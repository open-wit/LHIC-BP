"""
Encoding script 
Note: adapt model import paths if necessary.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from models.inference_lhic import LHIC_RNN_spectral, MSP_ARM, LSP_ARM
from coder.cbench.rans import BufferedRansEncoder


# ---------------------------
# Logging / deterministic / device
# ---------------------------

def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
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
# Patch / COT utils (must match decode)
# ---------------------------

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



def coding_order_table(patch_sz: int = 32) -> torch.Tensor:
    cot = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
    for i in range(patch_sz):
        start = i + 1
        cot[i, :] = torch.arange(start, start + patch_sz)
    return cot


def get_cot_sort_indices(patch_size: int = 32, device: torch.device | str = "cpu") -> torch.Tensor:
    cot = coding_order_table(patch_sz=patch_size).to(device)
    h_indices, w_indices = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    scaling_factor = patch_size**2 + 1
    compound_key = cot * scaling_factor + h_indices * patch_size + w_indices
    return torch.argsort(compound_key.view(-1))


def batch_sort_by_cot(target_batch: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """
    target_batch: (B,H,W) or (B,L)
    returns: (B,H*W) in COT order
    """
    device = target_batch.device
    sort_indices = get_cot_sort_indices(patch_size=patch_size, device=device)
    if target_batch.dim() == 3:
        flat = target_batch.view(target_batch.size(0), -1)
    else:
        flat = target_batch
    return flat[:, sort_indices]


# ---------------------------
# Data split utils
# ---------------------------

def get_sub_images(x_in: torch.Tensor, num_y_vals: int = 10000, param_d: int = 8):
    """
    x_in: (B,S,H,W) int32
    """
    x_msp_plane = (x_in >> param_d) << param_d
    x_lsp_ori = x_in - x_msp_plane
    x_lsp = x_lsp_ori.float() / (2**param_d - 1) * 2.0 - 1.0

    x_msp_plane = x_msp_plane.float() / float(num_y_vals)
    x_msp_plane = x_msp_plane * 2.0 - 1.0

    x_msp_ori = x_in >> param_d
    x_msp = x_msp_ori.float() / float(num_y_vals >> param_d)
    x_msp = x_msp * 2.0 - 1.0

    return x_lsp, x_lsp_ori, x_msp, x_msp_ori, x_msp_plane


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
    lsp_arm.load_state_dict(state_l, strict=True)
    lsp_arm = lsp_arm.to(device)

    msp_arm = MSP_ARM(cfg)
    state_m = torch.load(str(msp_ckpt_dir), map_location="cpu")
    msp_arm.load_state_dict(state_m, strict=True)
    msp_arm = msp_arm.to(device)

    # eval mode
    lsp_spec.eval()
    msp_spec.eval()
    lsp_arm.eval()
    msp_arm.eval()

    return lsp_spec, msp_spec, lsp_arm, msp_arm


# ---------------------------
# CDF helper
# ---------------------------

def modify_regular_prob(probs: torch.Tensor, precision: int = 16, num_y_val: int = 256) -> torch.Tensor:
    a = 2 ** (-precision)
    n = num_y_val
    return probs * (1 - n * a) + a

def compute_logistic_mixture_cdf(
    mus: torch.Tensor,
    scales: torch.Tensor,
    logits: torch.Tensor,
    precision: int = 16,
    bit_depth: int = 8,
    y_val: Optional[int] = None,
):
    """
    Returns cdfs_q (int32 quantized).
    """
    flag = False
    if mus.dim() == 4:
        b, h, w, _ = mus.shape
        flag = True
        mus = rearrange(mus, "b h w n -> b (h w) n")
        scales = rearrange(scales, "b h w n -> b (h w) n")
        logits = rearrange(logits, "b h w n -> b (h w) n")

    b, l, n = mus.shape
    max_v = (1 << bit_depth) - 1 if y_val is None else y_val
    size = max_v + 1
    interval = 1.0 / max_v

    endpoints = torch.arange(-1.0 + interval, 1.0, 2 * interval, device=mus.device).repeat((b, l, n, 1))

    mus = mus.unsqueeze(-1)
    scales = scales.unsqueeze(-1)
    pis = F.softmax(logits, dim=-1).unsqueeze(-1)
    invscale = torch.exp(-scales)

    rescaled = (endpoints - mus) * invscale
    cdfs = torch.zeros((b, l, n, size + 1), device=mus.device)
    cdfs[..., 1:-1] = torch.sigmoid(rescaled)
    cdfs[..., -1] = 1.0

    probs = cdfs[..., 1:] - cdfs[..., :-1]
    pmfs = (pis * probs).sum(dim=-2)

    pmfs = modify_regular_prob(pmfs, precision=precision, num_y_val=size)

    cdfs = torch.zeros((b, l, size + 1), device=mus.device)
    cdfs[..., 1:] = torch.cumsum(pmfs, dim=-1)
    cdfs[..., -1] = 1.0

    scale = float(1 << precision)
    cdfs_q = torch.round(cdfs * scale).to(torch.int32)

    if flag:
        cdfs_q = rearrange(cdfs_q, "b (h w) size -> b h w size", h=h, w=w)

    return cdfs_q


# ---------------------------
# Encoding
# ---------------------------

def encode_file(
    *,
    data_path: Path,
    cfg_path: Path,
    msp_ckpt_dir: Path,
    lsp_ckpt_dir: Path,
    out_path: Path,
    device: torch.device,
    patch_size: int = 32,
    param_d: int = 8,
    num_y_vals: int = 10000,
) -> None:
    torch.cuda.synchronize()
    t0 = time.time()

    logging.info(f"Loading data: {data_path}")
    arr = np.load(str(data_path))
    arr = np.clip(arr, 0, num_y_vals).astype(np.int32)

    x0 = torch.from_numpy(arr)  # (S,H,W)
    orig_shape = tuple(x0.shape)  # (S,H,W)

    x_in = img2patch(x0, patch_size)  # (B,S,ps,ps)
    patch_shape = tuple(x_in.shape)
    B, S, H, W = patch_shape
    assert H == patch_size and W == patch_size

    x_in = x_in.to(device)

    logging.info(f"orig_shape={orig_shape}, patch_shape={patch_shape}, device={device}")

    CFG = load_config(cfg_path)
    lsp_spec, msp_spec, lsp_arm, msp_arm = load_models(CFG, device, msp_ckpt_dir, lsp_ckpt_dir)
    
    # modules
    lsp_spatial_net = lsp_arm.spatial_net
    lsp_param_l0 = lsp_arm.param_l0
    lsp_param_net = lsp_arm.param_net
    lsp_context_net = lsp_arm.context_generate
    plane_net = lsp_arm.plane_net

    msp_spatial_net = msp_arm.spatial_net
    msp_param_l0 = msp_arm.param_l0
    msp_param_net = msp_arm.param_net
    msp_context_net = msp_arm.context_generate

    # preprocess
    x_lsp, x_lsp_ori, x_msp, x_msp_ori, x_msp_plane = get_sub_images(x_in, num_y_vals=num_y_vals, param_d=param_d)

    msp_encoder = BufferedRansEncoder()
    lsp_encoder = BufferedRansEncoder()

    logging.info("Start encoding...")
    torch.cuda.synchronize()
    t_enc0 = time.time()
    with torch.no_grad():
        x_plane_emb = plane_net.conv_in(x_msp_plane.unsqueeze(1))

    with torch.no_grad():
        msp_spec.reset_state()
        lsp_spec.reset_state()

        start_ch = torch.zeros(B, device=device) + 1  # MUST match decode

        for i in range(S):
            lsp_temp = x_lsp[:, i:i + 1]
            msp_temp = x_msp[:, i:i + 1]

            x_plane = plane_net.get_plane_prior(x_plane_emb[:, :, i])

            x_spatial_msp = msp_spatial_net(msp_temp)
            x_spatial_lsp = lsp_spatial_net(lsp_temp)

            if i == 0:
                param_lsp = lsp_param_l0(x_spatial_lsp, x_plane, return_list=False)
                param_msp = msp_param_l0(x_spatial_msp, return_list=False)
            else:
                x_spectral_lsp = lsp_spec(x_lsp[:, i - 1:i])
                x_spectral_lsp = lsp_context_net(x_spectral_lsp, start_ch)
                param_lsp = lsp_param_net(x_spectral_lsp, x_spatial_lsp, x_plane, return_list=False)

                x_spectral_msp = msp_spec(x_msp[:, i - 1:i])
                x_spectral_msp = msp_context_net(x_spectral_msp, start_ch)
                param_msp = msp_param_net(x_spectral_msp, x_spatial_msp, return_list=False)

                start_ch += 1

            # CDFs
            cdf_lsp = compute_logistic_mixture_cdf(
                param_lsp["mu"], param_lsp["scale"], param_lsp["weight"], bit_depth=param_d
            )
            cdf_msp = compute_logistic_mixture_cdf(
                param_msp["mu"], param_msp["scale"], param_msp["weight"], y_val=(num_y_vals >> param_d)
            )

            # samples (COT-ordered)
            lsp_sample = batch_sort_by_cot(x_lsp_ori[:, i].squeeze(1), patch_size=patch_size).T.contiguous().flatten().cpu().numpy()
            msp_sample = batch_sort_by_cot(x_msp_ori[:, i].squeeze(1), patch_size=patch_size).T.contiguous().flatten().cpu().numpy()

            # CDF reshape into list
            B0 = cdf_lsp.shape[0]

            cdf_lsp2 = rearrange(cdf_lsp, "b h w c -> (b c) h w")
            cdf_lsp2 = batch_sort_by_cot(cdf_lsp2, patch_size=patch_size)
            cdf_lsp2 = rearrange(cdf_lsp2, "(b c) l -> (l b) c", b=B0, c=(2**param_d + 1)).cpu().numpy()
            cdf_lsp_list = list(cdf_lsp2)

            cdf_msp2 = rearrange(cdf_msp, "b h w c -> (b c) h w")
            cdf_msp2 = batch_sort_by_cot(cdf_msp2, patch_size=patch_size)
            cdf_msp2 = rearrange(
                cdf_msp2, "(b c) l -> (l b) c", b=B0, c=((num_y_vals >> param_d) + 2)
            ).cpu().numpy()
            cdf_msp_list = list(cdf_msp2)

            indexes = np.arange(len(cdf_msp_list), dtype=np.int32)
            cdf_sizes_msp = np.array([len(cdf) for cdf in cdf_msp_list], dtype=np.int32)
            cdf_sizes_lsp = np.array([len(cdf) for cdf in cdf_lsp_list], dtype=np.int32)
            offsets = np.zeros(len(cdf_msp_list), dtype=np.int32)

            msp_encoder.encode_with_indexes_np(msp_sample, indexes, cdf_msp_list, cdf_sizes_msp, offsets)
            lsp_encoder.encode_with_indexes_np(lsp_sample, indexes, cdf_lsp_list, cdf_sizes_lsp, offsets)

    msp_bytes = msp_encoder.flush()
    lsp_bytes = lsp_encoder.flush()
    torch.cuda.synchronize()
    t_enc1 = time.time()

    # bits-per-pixel-per-band
    numel = int(x_in.numel())
    msp_bpp = len(msp_bytes) / numel * 8
    lsp_bpp = len(lsp_bytes) / numel * 8
    tot_bpp = (len(msp_bytes) + len(lsp_bytes)) / numel * 8

    logging.info(f"msp bpp={msp_bpp:.6f}, lsp bpp={lsp_bpp:.6f}, total bpp={tot_bpp:.6f}")
    logging.info(f"Encode compute time: {t_enc1 - t_enc0:.3f}s")

    code_res = [msp_bytes, lsp_bytes, x_in.shape, orig_shape, param_d, num_y_vals]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(code_res, f)

    logging.info(f"Saved: {out_path}")

# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to config.yml")
    p.add_argument("--msp_ckpt_dir", type=str, required=True, help="msp checkpoint dir")
    p.add_argument("--lsp_ckpt_dir", type=str, required=True, help="lsp checkpoint dir")
    p.add_argument("--data", type=str, required=True, help=".npy data file (S,H,W)")
    p.add_argument("--out", type=str, default=None, help="output .bin (pickle)")
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

    data_path = Path(args.data)
    out_path = Path(args.out) if args.out else data_path.with_suffix(".bin")

    encode_file(
        data_path=data_path,
        cfg_path=Path(args.config),
        msp_ckpt_dir=Path(args.msp_ckpt_dir),
        lsp_ckpt_dir=Path(args.lsp_ckpt_dir),
        out_path=out_path,
        device=device,
        patch_size=args.patch_size,
        param_d=args.param_d,
        num_y_vals=args.num_y_vals,
    )
