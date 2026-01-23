import gc
import types
import math



from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

import pytorch_lightning as pl

from modules.guided_diffusion import unet

nonlinearity = nn.SiLU

DEBUG_TIME=False

def __nop(ob):
    return ob
MyFunction = __nop


# Largely inspired by:https://github.com/thesofakillers/dlml-tutorial/blob/main/dlml.py
def compute_dlml_loss(
    means,
    log_scales,
    mixture_logits,
    y,
    output_min_bound=-1,
    output_max_bound=1,
    num_y_vals=128,
    reduction="mean",
):
    """
    Computes the Discretized Logistic Mixture Likelihood loss
    """
    inv_scales = torch.exp(-log_scales)

    y_range = output_max_bound - output_min_bound
    # explained in text
    epsilon = (0.5 * y_range) / (num_y_vals - 1)
    # convenience variable
    y = y.unsqueeze(-1).repeat(1, 1, 1, 1, means.shape[-1])
    centered_y = y - means
    # inputs to our sigmoid functions
    upper_bound_in = inv_scales * (centered_y + epsilon)
    lower_bound_in = inv_scales * (centered_y - epsilon)
    # remember: cdf of logistic distr is sigmoid of above input format
    upper_cdf = torch.sigmoid(upper_bound_in)
    lower_cdf = torch.sigmoid(lower_bound_in)
    # finally, the probability mass and equivalent log prob
    prob_mass = upper_cdf - lower_cdf

    # edges
    low_bound_log_prob = upper_bound_in - F.softplus(
        upper_bound_in
    )  # log probability for edge case of 0 (before scaling)
    upp_bound_log_prob = -F.softplus(
        lower_bound_in
    )  # log probability for edge case of 255 (before scaling)

    log_probs = torch.where(y < output_min_bound + 1e-6, low_bound_log_prob,
                            torch.where(y > output_max_bound - 1e-6, upp_bound_log_prob,
                                        torch.log(torch.clamp(prob_mass, min=1e-8))))

    # modeling which mixture to sample from
    log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1)

    # log likelihood
    log_likelihood = torch.sum(torch.logsumexp(log_probs,dim=-1,keepdim=True), dim=-1)

    prob = torch.exp(log_likelihood)
    a = float(2**(-16))
    prob = (1-num_y_vals*a)*prob +a
    loss = -torch.log2(prob)
    # print(loss)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    
    return loss

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x
    
class MaskedDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type="A", stride=1, padding=0):
        super(MaskedDWConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.mask_type = mask_type
        self.register_buffer("mask", torch.ones_like(self.depthwise.weight.data))
        _, _, h, w = self.mask.size()
        
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1 :, :] = 0
        self.depthwise.weight.data *= self.mask

    def forward(self, x):
        self.depthwise.weight.data *= self.mask
        x = self.depthwise(x)
        
        return x

def diagonal_dwconv_7x7_B(in_ch: int, out_ch: int) -> nn.Module:

    maskedconv = MaskedDWConv2d(in_ch, out_ch, kernel_size=7,mask_type='B', padding=3)

    maskedconv.mask[:, :, :, :] = 0
    for i in range(7):
        maskedconv.mask[:, :, i, :(7-i)] = 1

    return maskedconv

class DWConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type = "A", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1 :, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
def diagonal_conv_3x3(in_ch: int, out_ch: int) -> nn.Module:
    
    maskedconv = MaskedConv2d("A", in_ch, out_ch, kernel_size=3, padding=1)

    maskedconv.mask[:, :, 0, 2] = 0

    return maskedconv


class LHIC_RNN_spectral(nn.Module):
    def __init__(self, config, model_file, dim_enc, N_layers_spectral):
        super().__init__()

        self.config = config
        self.RUN_DEVICE = config.device
        self.state_bands = None

        self.dim_enc = dim_enc
        self.N_layers_spectral = N_layers_spectral

        self.conv_in = nn.Sequential(
            nn.Conv2d(1, dim_enc, kernel_size=1),
            nn.Conv2d(dim_enc, dim_enc, kernel_size=7, padding=3,groups=dim_enc),
            WSiLU()
            )
        
        self.spa_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_enc, dim_enc, kernel_size=7, padding=3,groups=dim_enc),
                nn.Conv2d(dim_enc, dim_enc, kernel_size=1),
            )
            for _ in range(N_layers_spectral)
        ])

        # load weights for spa and conv_in

        with torch.no_grad():
            wo = torch.load(model_file, map_location='cpu')
            w = {}
            
            # refine weights and send to correct device
            keys = list(wo.keys())

            for x in keys:
                if x.startswith('spectral_net.'):
                                
                    if '.time_' in x:
                        w[x[len('spectral_net.'):]] = wo[x].squeeze()
                        if DEBUG_TIME:
                            print(x, w[x[len('spectral_net.'):]].numpy())
                    if '.time_decay' in x:
                        w[x[len('spectral_net.'):]] = wo[x].float()
                        w[x[len('spectral_net.'):]] = -torch.exp(w[x[len('spectral_net.'):]])
                    elif '.time_first' in x:
                        w[x[len('spectral_net.'):]] = wo[x].float()
                    else:
                        w[x[len('spectral_net.'):]] = wo[x].float()

                    w[x[len('spectral_net.'):]].requires_grad = False
                    
                    if config.device == 'cuda' and x != 'emb.weight':
                        w[x[len('spectral_net.'):]] = w[x[len('spectral_net.'):]].to(config.device)

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        conv_in_weights = {
                k.replace("conv_in.", ""): v
                for k, v in w.items() 
                if k.startswith("conv_in.")
            }
        self.conv_in.load_state_dict(conv_in_weights,strict=True)
        for i in range(N_layers_spectral):
            spa_weights = {
                    k.replace(f"spectral_mix.{i}.spa.", ""): v
                    for k, v in w.items()
                    if k.startswith(f"spectral_mix.{i}.spa.")
                }
            self.spa_blocks[i].load_state_dict(spa_weights,strict=True)

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()


    def LN(self, x, w):
        return F.layer_norm(x, (self.dim_enc,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid(xr @ rw.T)
        k = torch.square(torch.relu(xk @ kw.T))
        kv = k @ vw.T

        return r * kv

    @MyFunction
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x

        r = torch.sigmoid(xr @ rw.T)
        k = xk @ kw.T
        v = xv @ vw.T

        kk = k
        vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = a / b
        
        return (r * wkv) @ ow.T 
    
    def reset_state(self):
        self.state_bands = None

    def forward(self, x):
        # x:(B,1,H,W) last band
        # B: batch size
        # C: number of columns
        # output: (BC,1,F) decodable prediction of line l for current band
        with torch.no_grad():
            w = self.w
            config = self.config
     
            B,_,H,W = x.shape

            delta = self.conv_in(x) # (B,F,H,W)

            delta = rearrange(delta, "b f h w -> (b h w) f")
            
            if self.state_bands == None:
                self.state_bands = torch.zeros(self.N_layers_spectral * 5, B*H*W, self.dim_enc, device=self.RUN_DEVICE)
                for i in range(self.N_layers_spectral):
                    self.state_bands[5*i+4] -= 1e30

            for i in range(self.N_layers_spectral):
                if i == 0:
                    delta = self.LN(delta, w.spectral_mix[i].ln0)
                
                ww = w.spectral_mix[i].att
                delta = delta + self.SA(self.LN(delta, w.spectral_mix[i].ln1), self.state_bands, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                ww = w.spectral_mix[i].ffn
                delta = delta + self.FF(self.LN(delta, w.spectral_mix[i].ln2), self.state_bands, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)

                delta = rearrange(delta, "1 (b h w) f -> b f h w",b=B,h=H,w=W)
                delta = delta + self.spa_blocks[i](delta)
                delta = rearrange(delta, "b f h w -> 1 (b h w) f")
                
            delta = rearrange(delta, "1 (b h w) f -> b f h w",b=B,h=H,w=W)

            return delta # (B,F,H,W)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

T_MAX = 512 # config.ctx_len
wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["./modules/cuda/wkv_op.cpp", "./modules/cuda/wkv_cuda.cu"], verbose=False, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert B * C % min(C, 32) == 0
        
        w = -torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        ctx.save_for_backward(w, u, k, v, y)
        
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert B * C % min(C, 32) == 0
        w, u, k, v, y = ctx.saved_tensors

        gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
        gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
        
        wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        return (None, None, None, gw, gu, gk, gv)


class Band_SpectralMix(nn.Module):
   
    def __init__(self, dim_enc, layer_id, n_layer):
        super().__init__()
        self.layer_id = layer_id
        self.dim_enc = dim_enc

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, dim_enc)
            for i in range(dim_enc):
                ddd[0, 0, i] = i / dim_enc
            
            # fancy time_decay
            decay_speed = torch.ones(dim_enc)
            for h in range(dim_enc):
                decay_speed[h] = -5 + 8 * (h / (dim_enc - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim_enc)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(dim_enc) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(dim_enc, dim_enc, bias=False)
        self.value = nn.Linear(dim_enc, dim_enc, bias=False)
        self.receptance = nn.Linear(dim_enc, dim_enc, bias=False)
        self.output = nn.Linear(dim_enc, dim_enc, bias=False)
        
    def jit_func(self, x):
        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, self.dim_enc, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)

class Band_ChannelMix(nn.Module):
    def __init__(self, dim_enc, layer_id, n_layer):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, dim_enc)
            for i in range(dim_enc):
                ddd[0, 0, i] = i / dim_enc
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0)) # mu-k
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0)) # mu-r
        
        self.key = nn.Linear(dim_enc, dim_enc, bias=False) # Wk
        self.receptance = nn.Linear(dim_enc, dim_enc, bias=False) # Wr
        self.value = nn.Linear(dim_enc, dim_enc, bias=False) #

    def forward(self, x):
        xx = self.time_shift(x) # y-2
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv



class Block(nn.Module):
    def __init__(self, dim_enc, layer_id, n_layers):
        super().__init__()

        self.layer_id = layer_id

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(dim_enc)
        
        self.ln1 = nn.LayerNorm(dim_enc)
        self.ln2 = nn.LayerNorm(dim_enc) 

        self.att = Band_SpectralMix(dim_enc, layer_id, n_layers)

        self.ffn = Band_ChannelMix(dim_enc, layer_id, n_layers)

        self.spa = nn.Sequential(
            nn.Conv2d(dim_enc, dim_enc, kernel_size=7, padding=3,groups=dim_enc),
            nn.Conv2d(dim_enc, dim_enc, kernel_size=1)
            )

    def forward(self, x): 
        '''
        b c f h w
        '''
        b, c ,f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        x = rearrange(x, '(b h w) c f -> (b c) f h w', h=h, w=w)
        x = x + self.spa(x)
        x = rearrange(x, '(b c) f h w -> b c f h w', c=c)
        # x = rearrange(x, '(b h w) c f -> b c f h w', h=h, w=w)

        return x

class RSCEM(nn.Module):
    def __init__(self, dim_spectral, n_layers):
        super().__init__()

        self.dim_spectral = dim_spectral

        self.conv_in = nn.Sequential(
            nn.Conv2d(1, dim_spectral, kernel_size=1),
            nn.Conv2d(dim_spectral, dim_spectral, kernel_size=7, padding=3,groups=dim_spectral),
            WSiLU()
            )

        self.spectral_mix = nn.ModuleList([Block(dim_spectral,i,n_layers) for i in range(n_layers)])

    def forward(self, x_shift:torch.Tensor) -> torch.Tensor:
        (B,C,H,W) = x_shift.shape

        x_shift = torch.reshape(x_shift,[B*C,1,H,W]) # B*C 1 H W

        x_shift = self.conv_in(x_shift) # B*C F H W

        x_shift = torch.reshape(x_shift,[B,C,self.dim_spectral,H,W])

        for _,block in enumerate(self.spectral_mix):
            x_shift = block(x_shift) # B C F H W

        return x_shift.reshape(B,C,self.dim_spectral,H,W) # B*C F H W

class MGCU(nn.Module):
    def __init__(self, dim_spatial):
        '''
        MGCU for spectral condition
        '''
        super().__init__()
        self.conv_in = nn.Conv2d(dim_spatial, dim_spatial, kernel_size=1)
        self.conv_1 = diagonal_dwconv_7x7_B(dim_spatial, dim_spatial)
        self.act = WSiLU()
        self.skip = nn.Conv2d(dim_spatial, dim_spatial, kernel_size=1)

        self.conv_out = nn.Conv2d(dim_spatial, dim_spatial, kernel_size=1)

        self.cache  = None
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        (B,F,C,P) = x.shape # B F C H*W
        x1 = self.conv_in(x)
        x1 = self.act(self.conv_1(x1))
        return self.conv_out(x1*self.skip(x))
    

class MGCU_s(nn.Module):
    def __init__(self, dim_spatial):
        '''
        MGCU for spatial context
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_spatial)
        self.MCG = MGCU(dim_spatial)
        self.norm2 = nn.LayerNorm(dim_spatial)
        self.convs = nn.Sequential(
            nn.Linear(dim_spatial, 4*dim_spatial),
            WSiLU(),
            nn.Linear(4*dim_spatial, dim_spatial)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        (P,F,H,W) = x.shape # B*C F H W
        x1 = self.norm1(torch.reshape(torch.permute(x, (0,2,3,1)),[P,H*W,F])).contiguous() # B*C,H*W,FF
        x1 = torch.permute(torch.reshape(x1,[P,H,W,F]),(0,3,1,2)).contiguous() # B*C F H W
        x1 = self.MCG(x1)
        x1 = x+x1
        x2 = self.norm2(torch.reshape(torch.permute(x1, (0,2,3,1)),[P,H*W,F])).contiguous() # B*C,H*W,FF
        x2 = torch.permute(torch.reshape(self.convs(x2),[P,H,W,F]),(0,3,1,2)).contiguous()
        return x2+x1 # B*C F H W


class PSCAM(nn.Module):
    '''
    parallel spatial context
    '''
    def __init__(self, dim_spatial,N_layers_spatial):
        super(PSCAM, self).__init__()

        self.spatial_mix = nn.ModuleList([
            MGCU_s(dim_spatial)
            for _ in range(N_layers_spatial)
        ])

        self.conv_in = diagonal_conv_3x3(1,dim_spatial)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        x:B,C,H,W
        '''
        (B,C,H,W) = x.shape
        x = rearrange(x, 'b c h w -> (b c) 1 h w')

        x = self.conv_in(x)

        for _,block in enumerate(self.spatial_mix):
            x = block(x)

        return x #B*C,F1,H,W


class Band_SpatialMix(nn.Module):
    '''
    hold context and generate
    '''
    def __init__(self, dim_spectral, N_layers_context, dropout):
        super(Band_SpatialMix, self).__init__()
        self.spatial_mix = unet.UNetModel(dim_spectral,
                                    dim_spectral,
                                    N_layers_context,
                                    (),
                                    dropout,
                                    tuple([1]))

    def forward(self, x_spectral, start_ch)->torch.Tensor:
        '''
        train mode
        size:B,C,H,W
        '''
        if x_spectral.dim()==4:
            (B,FF,H,W) = x_spectral.shape
            x_param = self.spatial_mix(x_spectral,start_ch) # B*C F H W
            return x_param
        else:
            (B,C,FF,H,W) = x_spectral.shape
            x_spectral = x_spectral.reshape(B*C,-1,H,W)
            t_base = start_ch
            timesteps = torch.cat([t_base+i for i in range(1,C+1)])
            x_param = self.spatial_mix(x_spectral,timesteps) # B*C F H W
            return x_param.reshape(B,C,FF,H,W)
    
class Plane_Context(nn.Module):
    '''
    plane context for lsp
    '''
    def __init__(self, config):
        super(Plane_Context, self).__init__()
        self.config = config
        self.conv_in = DWConv3D(in_channels=1,out_channels=config.dim_plane,kernel_size=(7,7,7),stride=1,padding=3)
        self.plane_context = unet.UNetModel(config.dim_plane,
                                    config.dim_plane,
                                    config.N_layers_plane,
                                    (),
                                    config.dropout,
                                    tuple([1]))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        (B,C,H,W) = x.shape
        x1 = x.reshape(B,1,C,H,W)
        x1 = self.conv_in(x1) # B,F,C,H,W
        x_plane = torch.zeros(B,C,self.config.dim_plane,H,W,device=x.device)
        t_base = torch.ones(B, device=x.device)
        timesteps = torch.cat([t_base for _ in range(1,C+1)])
        x_plane = self.plane_context(torch.reshape(torch.permute(x1,[0,2,1,3,4]),[B*C,-1,H,W]),timesteps)
        return x_plane.reshape(B,C,-1,H,W)
        
    def get_plane_prior(self, x_emb:torch.Tensor):
        '''
        use for inference
        x_emb:B,F,H,W
        '''
        (B,C,H,W) = x_emb.shape
        t_base = torch.ones(B, device=x_emb.device)
        x_plane = self.plane_context(x_emb,torch.cat([t_base]))

        return x_plane

class ParamBlock(nn.Module):
    """
    A residual block for param estimate
    """

    def __init__(
        self,
        channels,
        out_channels
    ):
        super().__init__()
        self.channels = channels

        self.in_layers = nn.Sequential(
            nn.SiLU(),
            conv1x1(channels, out_channels, 1),
        )

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            conv1x1(out_channels, out_channels, 1)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)

        return x + h


class Param_Net(nn.Module):
    def __init__(self,dim_in,dim_params,config):
        super(Param_Net, self).__init__()
        self.config = config
        self.conv_in = conv1x1(dim_in,dim_params)
        self.blocks = nn.ModuleList([ParamBlock(dim_params,dim_params) for _ in range(config.N_layers_params)])
        self.conv_out_mu = conv1x1(dim_params, config.num_mixture)
        self.conv_out_scale = conv1x1(dim_params, config.num_mixture)
        self.act = WSiLU()
        self.conv_out_weight = conv1x1(dim_params, config.num_mixture)
        
    def forward(self, x_spectral:torch.Tensor, x_spatial:torch.Tensor, x_plane=None, return_list=True)->dict:
        '''
        input must be (B,C,FF,H,W)
        this function use for combine spectral and spatial context,then generate the mixture parameters
        '''
        features = [x_spatial,x_spectral]
        if x_plane is not None:
            features = [x_plane] + features
        
        if x_spectral.dim() == 4:
            (B,FF,H,W)=x_spatial.shape
            x_in = torch.cat(features,dim=1)
        else:
            (B,C,FF,H,W)=x_spatial.shape
            x_in = torch.cat(features,dim=1).reshape(B*C,-1,H,W)
            
        x_in = rearrange(x_in, 'b c h w -> b c (h w) 1')
        x_out = self.conv_in(x_in)

        for _,block in enumerate(self.blocks):
            x_out = block(x_out)

        x_out = self.act(x_out)
        
        x_out_mu = self.conv_out_mu(x_out) # 
        x_out_scale = self.conv_out_scale(x_out) # 
        x_out_weight = self.conv_out_weight(x_out) # BC N H W
        
        if return_list:
            x_out_mu = rearrange(x_out_mu, 'b c h w -> b h w c')
            x_out_scale = rearrange(x_out_scale, 'b c h w -> b h w c')
            x_out_weight = rearrange(x_out_weight, 'b c h w -> b h w c')
        else:
            x_out_mu = rearrange(x_out_mu, 'b c (h w) 1 -> b h w c',h=H,w=W)
            x_out_scale = rearrange(x_out_scale, 'b c (h w) 1 -> b h w c',h=H,w=W)
            x_out_weight = rearrange(x_out_weight, 'b c (h w) 1 -> b h w c',h=H,w=W)
        
        return {'mu':x_out_mu, 'scale':x_out_scale, 'weight':x_out_weight}
        

class Param_L0(nn.Module):
    def __init__(self, dim_in, dim_params,config):
        super(Param_L0, self).__init__()
        self.config = config
        self.conv_in = None
        if dim_in != dim_params:
            self.conv_in = conv1x1(dim_in,dim_params)
        self.block = ParamBlock(dim_params,dim_params)
        self.conv_out_mu = conv1x1(dim_params, config.num_mixture)
        self.conv_out_scale = conv1x1(dim_params, config.num_mixture)
        self.act = WSiLU()
        if config.num_mixture > 1:
            self.conv_out_weight = conv1x1(dim_params, config.num_mixture)
        

    def forward(self, x_spatial:torch.Tensor, x_plane=None, return_list=False)->dict:
        '''
        input must be (B,FF,H,W)
        this function use for combine spectral and spatial context,then generate the mixture parameters
        '''
        (P,FF,H,W)=x_spatial.shape
        if x_plane is not None:
            x_in = torch.concat([x_plane, x_spatial],dim=1)
        
        else:
            x_in = x_spatial # (B*C,F1+F2,H,W)
        
        x_in = rearrange(x_in, 'b c h w -> b c (h w) 1')
        if self.conv_in is not None:
            x_in = self.conv_in(x_in)
        x_in = self.block(x_in)

        x_out = self.act(x_in)
        
        x_out_mu = self.conv_out_mu(x_out) # 
        x_out_scale = self.conv_out_scale(x_out) # 
        x_out_weight = self.conv_out_weight(x_out) # BC 5 H W
        if return_list:
            x_out_mu = rearrange(x_out_mu, 'b c h w -> b h w c')
            x_out_scale = rearrange(x_out_scale, 'b c h w -> b h w c')
            x_out_weight = rearrange(x_out_weight, 'b c h w -> b h w c')
        else:
            x_out_mu = rearrange(x_out_mu, 'b c (h w) 1 -> b h w c',h=H,w=W)
            x_out_scale = rearrange(x_out_scale, 'b c (h w) 1 -> b h w c',h=H,w=W)
            x_out_weight = rearrange(x_out_weight, 'b c (h w) 1 -> b h w c',h=H,w=W)
        
        return {'mu':x_out_mu, 'scale':x_out_scale, 'weight':x_out_weight}
        

    
class Base_Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config=config
    
    def configure_optimizers(self):
        """layerwise learning rate"""
        config = self.config
        
        assert hasattr(config, 'layerwise_lr'), "config lack layerwise_lr parameter"
        assert hasattr(config, 'weight_decay'), "config lack weight_decay parameter"
        
        # init groups
        groups = {
            'lr_decay': set(),  # param for weight decay
            'lr_1x': set(),     # foundational learning rate
            'lr_2x': set(),     # twice learning rate
            'lr_3x': set()      # three times learning rate
        }
        
        # parameters group
        for name, param in self.named_parameters():
            if config.layerwise_lr > 0:
                if "time_mix" in name or "time_faaaa" in name:
                    groups['lr_1x'].add(name)
                elif "time_decay" in name:
                    groups['lr_2x'].add(name)
                elif "time_first" in name:
                    groups['lr_3x'].add(name)
                elif len(param.squeeze().shape) >= 2 and config.weight_decay > 0:
                    groups['lr_decay'].add(name)
                else:
                    groups['lr_1x'].add(name)
            else:
                groups['lr_1x'].add(name)
        
        param_dict = dict(self.named_parameters())
        optim_groups = []
        
        # build optimizers
        if config.layerwise_lr > 0:
            optim_groups.extend([
                {"params": [param_dict[n] for n in sorted(groups['lr_1x'])], 
                "weight_decay": 0.0, "lr_scale": 1.0},
                {"params": [param_dict[n] for n in sorted(groups['lr_2x'])],
                "weight_decay": 0.0, "lr_scale": 2.0},
                {"params": [param_dict[n] for n in sorted(groups['lr_3x'])],
                "weight_decay": 0.0, "lr_scale": 3.0}
            ])
        
        if config.weight_decay > 0 and groups['lr_decay']:
            optim_groups.append({
                "params": [param_dict[n] for n in sorted(groups['lr_decay'])],
                "weight_decay": config.weight_decay,
                "lr_scale": 1.0
            })
        
        return torch.optim.AdamW(
            optim_groups,
            lr=config.lr_init,
            weight_decay=0,  # 
            amsgrad=False
        )

class MSP_ARM(Base_Model):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.config = config
        self.sig_num = config.sig_num
        self.ins_num = config.bit_depth - config.sig_num
        self.spectral_net = RSCEM(config.dim_msp_spectral,config.N_layers_spectral_msp)
        self.context_generate = Band_SpatialMix(config.dim_msp_spectral, config.N_layers_context, config.dropout)
        self.spatial_net = PSCAM(config.dim_msp_spatial,config.N_layers_spatial_msp)
        self.param_net = Param_Net(config.dim_msp_spatial+config.dim_msp_spectral,config.dim_params,config)
        self.param_l0 = Param_L0(config.dim_msp_spatial, config.dim_params,config)
    def forward(self,x,start_ch=None):
        pass
    
class LSP_ARM(Base_Model):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.ins_num = self.config.bit_depth - self.config.sig_num
        self.plane_net = Plane_Context(config)
        self.spatial_net = PSCAM(config.dim_lsp_spatial,config.N_layers_spatial_lsp)
        self.spectral_net = RSCEM(config.dim_lsp_spectral,config.N_layers_spectral_lsp)
        self.context_generate = Band_SpatialMix(config.dim_lsp_spectral, config.N_layers_context, config.dropout)
        self.param_net = Param_Net(config.dim_lsp_spatial+config.dim_lsp_spectral+config.dim_plane,config.dim_params, config)
        self.param_l0 = Param_L0(config.dim_lsp_spatial+config.dim_plane,config.dim_params, config)

    def forward(self,x,start_ch=None):
        pass