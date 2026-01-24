"""
Decoding script
Note: adapt model import paths if necessary.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F

from models.inference_lhic import LHIC_RNN_spectral, MSP_ARM, LSP_ARM
from coder.cbench.rans import RansDecoder


# ---------------------------
# Logging / deterministic / device
# ---------------------------

def setup_logger(verbose: bool = False) -> None:
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def set_deterministic(deterministic: bool = True) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def setup_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = device_str.lower()
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        logging.warning("CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------
# Patch / COT utils (match encoder)
# ---------------------------
def modify_regular_prob(
        probs:torch.Tensor,
        precision=16,
        num_y_val=256,
        )->np.ndarray:
    '''
        function to modify the frequencies of the symbols that all possible value has at lease 1 frequency
    '''
    a = 2 ** (-precision)
    n = num_y_val
    probs = probs*(1-n*a)+a
    return probs


def compute_logistic_mixture_cdf(
    mus: torch.Tensor, 
    scales: torch.Tensor, 
    logits: torch.Tensor, 
    precision=16,
    bit_depth=8,
    y_val=None
):
    """
    compute Logistic Mixture CDF 
    
    """
    # reshape input to b 1 num_mix
    flag=False
    if mus.dim() == 4:
        b,h,w,_ = mus.shape
        flag = True
        mus = rearrange(mus, 'b h w n -> b (h w) n')
        scales = rearrange(scales, 'b h w n -> b (h w) n')
        logits = rearrange(logits, 'b h w n -> b (h w) n')

    b, l, n = mus.shape
    if y_val is None:
        max_v = (1 << bit_depth) - 1
    else:
        max_v = y_val
    size = max_v + 1
    interval = 1. / max_v
    endpoints = torch.arange(-1.0+interval,1.0, 2*interval, device=mus.device).repeat((b,l,n,1)) # [b, l, num_mix, max_v]

    # reshape parameters
    mus = mus.unsqueeze(-1)        # [b, l, num_mix, 1]
    scales = scales.unsqueeze(-1)  # [b, l, num_mix, 1]
    pis = F.softmax(logits, dim=-1).unsqueeze(-1) # [b, l, num_mix, 1]
    invscale = torch.exp(-scales)

    # logistic mixture CDFs: [b, l, n, max_v]
    rescaled = (endpoints - mus) * invscale
    cdfs = torch.zeros((b, l, n, size+1), device=mus.device)
    cdfs[...,1:-1] = torch.sigmoid(rescaled)
    cdfs[...,-1] = 1.0

    probs = cdfs[..., 1:] - cdfs[..., :-1] # [b, l, n, size]
    pmfs = (pis * probs).sum(dim=-2) # [b, l, size]

    pmfs = modify_regular_prob(pmfs,precision=precision,num_y_val=size)

    cdfs = torch.zeros((b, l, size+1), device=mus.device)
    cdfs[..., 1:] = torch.cumsum(pmfs, dim=-1)
    cdfs[..., -1] = 1.0

    scale = float(1 << precision)
    cdfs_q = torch.round(cdfs * scale).to(torch.int32)

    if flag:
    # return b h w size
        pmfs = rearrange(pmfs, 'b (h w) size -> b h w size', h=h, w=w)
        cdfs_q = rearrange(cdfs_q, 'b (h w) size -> b h w size', h=h, w=w)


    return pmfs, cdfs_q

def img2patch(img: torch.Tensor, patch_sz: int) -> torch.Tensor:
    """
    (C,H,W) -> (B,C,patch_sz,patch_sz)
    Pad RIGHT/BOTTOM, patch order: hi-major then wi (matches nested loops).
    """
    C, H, W = img.shape
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    img_pad = F.pad(img, (0, pad_w, 0, pad_h))  # right + bottom

    H2, W2 = img_pad.shape[-2], img_pad.shape[-1]
    h_num = H2 // patch_sz
    w_num = W2 // patch_sz

    x = img_pad.view(C, h_num, patch_sz, w_num, patch_sz)
    x = x.permute(1, 3, 0, 2, 4).contiguous()  # (h_num,w_num,C,ps,ps)
    return x.view(h_num * w_num, C, patch_sz, patch_sz)


def patch2img(patches: torch.Tensor, img_h: int, img_w: int, patch_sz: int) -> torch.Tensor:
    """
    Inverse of img2patch above.
    (B,C,patch_sz,patch_sz) -> (C,img_h,img_w), crops RIGHT/BOTTOM padding.
    """
    B, C, _, _ = patches.shape
    h_num = math.ceil(img_h / patch_sz)
    w_num = math.ceil(img_w / patch_sz)

    grid = patches.view(h_num, w_num, C, patch_sz, patch_sz)  # (h_num,w_num,C,ps,ps)
    img_pad = grid.permute(2, 0, 3, 1, 4).contiguous()        # (C,h_num,ps,w_num,ps)
    img_pad = img_pad.view(C, h_num * patch_sz, w_num * patch_sz)
    return img_pad[:, :img_h, :img_w]


def coding_order_table(patch_sz: int = 32) -> torch.Tensor:
    cot = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
    for i in range(patch_sz):
        start = i + 1
        cot[i, :] = torch.arange(start, start + patch_sz)
    return cot


@dataclass
class StepIndex:
    h_cpu: torch.Tensor
    w_cpu: torch.Tensor
    h_dev: torch.Tensor
    w_dev: torch.Tensor

    @property
    def k(self) -> int:
        return int(self.h_cpu.numel())


def build_cot_step_indices(patch_sz: int, device: torch.device) -> List[StepIndex]:
    """Precompute indices for each COT step j (1..max)."""
    cot_cpu = coding_order_table(patch_sz=patch_sz)  # CPU
    max_step = int(cot_cpu.max().item())
    steps: List[StepIndex] = []

    for j in range(1, max_step + 1):
        h_cpu, w_cpu = torch.nonzero(cot_cpu == j, as_tuple=True)
        steps.append(
            StepIndex(
                h_cpu=h_cpu.long(),
                w_cpu=w_cpu.long(),
                h_dev=h_cpu.long().to(device),
                w_dev=w_cpu.long().to(device),
            )
        )
    return steps


# ---------------------------
# Config / model loaders
# ---------------------------

def load_config(config_path: Path):
    return OmegaConf.load(config_path)


def load_models(cfg, device: torch.device, msp_ckpt_dir: Path, lsp_ckpt_dir: Path):
    # spectral (stateful)
    lsp_spec = LHIC_RNN_spectral(
        cfg,
        str(lsp_ckpt_dir),
        cfg.dim_lsp_spectral,
        cfg.N_layers_spectral_lsp,
    ).to(device)
    msp_spec = LHIC_RNN_spectral(
        cfg,
        str(msp_ckpt_dir),
        cfg.dim_msp_spectral,
        cfg.N_layers_spectral_msp,
    ).to(device)

    # ARM
    lsp_arm = LSP_ARM(cfg)
    state_l = torch.load(str(lsp_ckpt_dir), map_location="cpu")
    lsp_arm.load_state_dict(state_l)
    lsp_arm = lsp_arm.to(device)

    msp_arm = MSP_ARM(cfg)
    state_m = torch.load(str(msp_ckpt_dir), map_location="cpu")
    msp_arm.load_state_dict(state_m)
    msp_arm = msp_arm.to(device)

    # eval mode
    lsp_spec.eval()
    msp_spec.eval()
    lsp_arm.eval()
    msp_arm.eval()

    return lsp_spec, msp_spec, lsp_arm, msp_arm


# ---------------------------
# RANS step decode helper
# ---------------------------

def rans_decode_block(
    decoder: RansDecoder,
    cdf: torch.Tensor,
    B: int,
    k: int,
) -> torch.Tensor:
    """
    cdf: (B, k, N) int-ish tensor
    returns: (B, k) int32 tensor on device of cdf
    """
    device = cdf.device
    if cdf.dim() == 4:
        # e.g. (B,k,1,N)
        cdf = cdf.squeeze(2)
    assert cdf.dim() == 3, f"Unexpected cdf shape: {tuple(cdf.shape)}"

    # flatten as (k*B, N) in (l b) order
    cdf_np = cdf.permute(1, 0, 2).contiguous().view(-1, cdf.size(-1))
    cdf_np = cdf_np.detach().cpu().numpy().astype(np.int32, copy=False)
    cdf_list = list(cdf_np)  # list of 1D arrays

    n_syms = len(cdf_list)
    indexes = np.arange(n_syms, dtype=np.int32)
    cdf_sizes = np.full((n_syms,), cdf_np.shape[1], dtype=np.int32)
    offsets = np.zeros((n_syms,), dtype=np.int32)

    syms = decoder.decode_stream_np(indexes, cdf_list, cdf_sizes, offsets)
    syms = np.asarray(syms, dtype=np.int32)  # (k*B,)
    syms_t = torch.from_numpy(syms).to(device=device)

    # reshape back (k,B) -> (B,k)
    syms_t = syms_t.view(k, B).permute(1, 0).contiguous()
    return syms_t


# ---------------------------
# Decode MSP / LSP
# ---------------------------

def decode_msp(
    *,
    decoder: RansDecoder,
    msp_spectral_net,
    msp_spatial_net,
    msp_param_l0,
    msp_param_net,
    msp_context_net,
    steps: List[StepIndex],
    B: int,
    S: int,
    H: int,
    W: int,
    device: torch.device,
    param_d: int,
    num_y_vals: int,
) -> torch.Tensor:
    """
    Returns msp_ori: (B,S,H,W) int32 on CPU
    """
    msp_ori_cpu = torch.zeros((B, S, H, W), dtype=torch.int32, device="cpu")
    logging.info(f"start decoding MSP")
    with torch.no_grad():
        msp_spectral_net.reset_state()

        start_ch = torch.zeros((B,), device=device) + 1  # MUST match encoder
        prev_norm = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for i in range(S):
            if i > 0:
                x_spectral = msp_spectral_net(prev_norm)
                x_spectral = msp_context_net(x_spectral, start_ch)
                start_ch += 1
            else:
                x_spectral = None

            curr_norm = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

            for step in steps:
                h, w = step.h_dev, step.w_dev
                k = step.k

                x_spatial = msp_spatial_net(curr_norm)
                if i == 0:
                    param = msp_param_l0(x_spatial, return_list=False)
                else:
                    param = msp_param_net(x_spectral, x_spatial, return_list=False)

                mu = param["mu"][:, h, w]
                scale = param["scale"][:, h, w]
                weight = param["weight"][:, h, w]

                _, cdf = compute_logistic_mixture_cdf(mu, scale, weight, y_val=(num_y_vals >> param_d))
                sample = rans_decode_block(decoder, cdf, B=B, k=k)  # (B,k)

                # write int symbols
                msp_ori_cpu[:, i, step.h_cpu, step.w_cpu] = sample.detach().cpu()

                # update norm
                curr_norm[:, 0, h, w] = sample.float() / float(num_y_vals >> param_d) * 2.0 - 1.0

            prev_norm = curr_norm

    return msp_ori_cpu


def decode_lsp(
    *,
    decoder: RansDecoder,
    lsp_spectral_net,
    lsp_spatial_net,
    lsp_param_l0,
    lsp_param_net,
    lsp_context_net,
    plane_net,
    x_plane_emb: torch.Tensor,  # (B, ?, S, H, W)
    steps: List[StepIndex],
    B: int,
    S: int,
    H: int,
    W: int,
    device: torch.device,
    param_d: int,
) -> torch.Tensor:
    """
    Returns lsp_ori: (B,S,H,W) int32 on CPU
    """
    lsp_ori_cpu = torch.zeros((B, S, H, W), dtype=torch.int32, device="cpu")
    logging.info(f"start decoding LSP")
    with torch.no_grad():
        lsp_spectral_net.reset_state()

        start_ch = torch.zeros((B,), device=device) + 1  # MUST match encoder
        prev_norm = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

        for i in range(S):
            # plane prior for this channel
            x_emb_i = x_plane_emb[:, :, i]
            x_plane = plane_net.get_plane_prior(x_emb_i)

            if i > 0:
                x_spectral = lsp_spectral_net(prev_norm)
                x_spectral = lsp_context_net(x_spectral, start_ch)
                start_ch += 1
            else:
                x_spectral = None

            curr_norm = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

            for step in steps:
                h, w = step.h_dev, step.w_dev
                k = step.k

                x_spatial = lsp_spatial_net(curr_norm)
                if i == 0:
                    param = lsp_param_l0(x_spatial, x_plane, return_list=False)
                else:
                    param = lsp_param_net(x_spectral, x_spatial, x_plane, return_list=False)

                mu = param["mu"][:, h, w]
                scale = param["scale"][:, h, w]
                weight = param["weight"][:, h, w]

                _, cdf = compute_logistic_mixture_cdf(mu, scale, weight, bit_depth=param_d)
                sample = rans_decode_block(decoder, cdf, B=B, k=k)  # (B,k)

                lsp_ori_cpu[:, i, step.h_cpu, step.w_cpu] = sample.detach().cpu()
                curr_norm[:, 0, h, w] = sample.float() / float((2**param_d) - 1) * 2.0 - 1.0

            prev_norm = curr_norm

    return lsp_ori_cpu


# ---------------------------
# Main decode
# ---------------------------

def decode_file(
    *,
    bin_path: Path,
    cfg_path: Path,
    msp_ckpt_dir: Path,
    lsp_ckpt_dir: Path,
    out_path: Path,
    device: torch.device,
    patch_size: int,
    param_d: int,
    num_y_vals: int,
    data_path: Optional[Path] = None,
) -> None:
    torch.cuda.synchronize()
    t0 = time.time()

    logging.info(f"Loading bitstream: {bin_path}")
    with open(bin_path, "rb") as f:
        msp_bytes, lsp_bytes, patch_shape, orig_shape, param_d, num_y_vals = pickle.load(f)

    patch_shape = tuple(patch_shape)
    if len(patch_shape) != 4:
        raise ValueError(f"Invalid patch_shape in bin: {patch_shape}")
    B, S, H, W = patch_shape

    if H != patch_size or W != patch_size:
        raise ValueError(f"patch_size mismatch: bin has (H,W)=({H},{W}), args patch_size={patch_size}")

    logging.info(f"Patched shape from bin: B={B}, S={S}, H={H}, W={W}")
    logging.info(f"Loading config/models: {cfg_path} / {msp_ckpt_dir} / {lsp_ckpt_dir}")

    CFG = load_config(cfg_path)
    lsp_spec, msp_spec, lsp_arm, msp_arm = load_models(CFG, device, msp_ckpt_dir, lsp_ckpt_dir)

    # modules
    # MSP
    msp_spatial_net = msp_arm.spatial_net
    msp_param_l0 = msp_arm.param_l0
    msp_param_net = msp_arm.param_net
    msp_context_net = msp_arm.context_generate

    # LSP
    lsp_spatial_net = lsp_arm.spatial_net
    lsp_param_l0 = lsp_arm.param_l0
    lsp_param_net = lsp_arm.param_net
    lsp_context_net = lsp_arm.context_generate
    plane_net = lsp_arm.plane_net

    # COT steps
    steps = build_cot_step_indices(patch_sz=patch_size, device=device)
    logging.info(f"COT steps: {len(steps)} (max_step)")

    # Decode MSP
    torch.cuda.synchronize()
    t_msp0 = time.time()
    msp_dec = RansDecoder()
    msp_dec.set_stream(msp_bytes)

    msp_ori_cpu = decode_msp(
        decoder=msp_dec,
        msp_spectral_net=msp_spec,
        msp_spatial_net=msp_spatial_net,
        msp_param_l0=msp_param_l0,
        msp_param_net=msp_param_net,
        msp_context_net=msp_context_net,
        steps=steps,
        B=B, S=S, H=H, W=W,
        device=device,
        param_d=param_d,
        num_y_vals=num_y_vals,
    )
    torch.cuda.synchronize()
    t_msp1 = time.time()
    logging.info(f"MSP decode done in {t_msp1 - t_msp0:.3f}s")

    # Build plane embedding from decoded MSP
    logging.info("Building plane embedding for LSP...")
    with torch.no_grad():
        msp_plane = (msp_ori_cpu.to(device) << param_d).float() / float(num_y_vals)
        msp_plane = msp_plane * 2.0 - 1.0  # (B,S,H,W)
        x_plane_emb = plane_net.conv_in(msp_plane.unsqueeze(1))  # (B,?,S,H,W) per your model

    # Decode LSP
    torch.cuda.synchronize()
    t_lsp0 = time.time()
    lsp_dec = RansDecoder()
    lsp_dec.set_stream(lsp_bytes)

    lsp_ori_cpu = decode_lsp(
        decoder=lsp_dec,
        lsp_spectral_net=lsp_spec,
        lsp_spatial_net=lsp_spatial_net,
        lsp_param_l0=lsp_param_l0,
        lsp_param_net=lsp_param_net,
        lsp_context_net=lsp_context_net,
        plane_net=plane_net,
        x_plane_emb=x_plane_emb,
        steps=steps,
        B=B, S=S, H=H, W=W,
        device=device,
        param_d=param_d,
    )
    torch.cuda.synchronize()
    t_lsp1 = time.time()
    logging.info(f"LSP decode done in {t_lsp1 - t_lsp0:.3f}s")

    # Reconstruct x_in patches
    x_hat = (msp_ori_cpu.to(torch.int32) << param_d) + lsp_ori_cpu.to(torch.int32)
    x_hat = torch.clamp(x_hat, 0, num_y_vals).to(torch.int32)
    x_hat = patch2img(x_hat, orig_shape[1], orig_shape[2], patch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), x_hat.numpy())
    logging.info(f"Saved decoded patches: {out_path}")

    # Optional verification
    if data_path is not None:
        logging.info(f"Verifying with original data: {data_path}")
        arr = np.load(str(data_path))
        arr = np.clip(arr, 0, num_y_vals).astype(np.int32)
        x_in = torch.from_numpy(arr).to(torch.int32)
        # x_in = img2patch(x_in, patch_size)
        if tuple(x_in.shape) != orig_shape:
            logging.warning(f"Original patched shape {tuple(x_in.shape)} != decoded shape {orig_shape}")
        diff = (x_hat.cpu() - x_in.cpu()).numpy()
        neq = np.mean(diff != 0)
        logging.info(f"Verify: mismatch ratio = {neq:.6f}, diff min={diff.min()}, max={diff.max()}")
    torch.cuda.synchronize()
    t1 = time.time()
    logging.info(f"Total decode time: {t1 - t0:.3f}s")


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to config.yml")
    p.add_argument("--msp_ckpt_dir", type=str, required=True, help="msp checkpoint dir")
    p.add_argument("--lsp_ckpt_dir", type=str, required=True, help="lsp checkpoint dir")
    p.add_argument("--bin", type=str, required=True, help="encoded .bin file (pickle)")
    p.add_argument("--out", type=str, default=None, help="output .npy (decoded patches)")
    p.add_argument("--data", type=str, default=None, help="optional original .npy for verification")
    p.add_argument("--device", type=str, default='cuda', help="cpu / cuda / cuda:0 ... (default auto)")
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--param_d", type=int, default=8)
    p.add_argument("--num_y_vals", type=int, default=10000)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    setup_logger()
    set_deterministic(True)

    device = setup_device(args.device)
    logging.info(f"Using device: {device}")

    bin_path = Path(args.bin)
    out_path = Path(args.out) if args.out is not None else bin_path.with_suffix(".decoded_files.npy")

    decode_file(
        bin_path=bin_path,
        cfg_path=Path(args.config),
        msp_ckpt_dir=Path(args.msp_ckpt_dir),
        lsp_ckpt_dir=Path(args.lsp_ckpt_dir),
        out_path=out_path,
        device=device,
        patch_size=args.patch_size,
        param_d=args.param_d,
        num_y_vals=args.num_y_vals,
        data_path=Path(args.data) if args.data else None,
    )
