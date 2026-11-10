"""Turn a stemd_roformer training checkpoint into a standard BS-RoFormer checkpoint.

stemd_roformer is a 4-stem BS-RoFormer (bass, drums, other, vocals) trained with its
own module names. This renames its tensors to ZFTurbo's `BSRoformer` names, which
are the names `crates/stemd-mlx/src/roformer.rs` reads, and writes the matching
config. `convert_roformer.py` then checks it and writes the safetensors:

    python tools/export/convert_stemd_roformer.py --ckpt gt_982000.pt ... gt_1000000.pt \\
        --out stemd_roformer.ckpt
    python tools/export/convert_roformer.py --ckpt stemd_roformer.ckpt \\
        --config stemd_roformer.yaml --out stemd_roformer.safetensors

Several `--ckpt` are averaged weight by weight. A checkpoint saved with EMA
weights holds them under `model` like any other.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import yaml

DIM, DEPTH, HEADS, DIM_HEAD = 256, 12, 8, 64
STEMS = ["bass", "drums", "other", "vocals"]
# A tuple, as the reference type-checks it; `yaml.dump` writes it as !!python/tuple.
FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)

RENAMES = [
    (r"band_norm\.(\d+)\.gamma", r"band_split.to_features.\1.0.gamma"),
    (r"band_lin\.(\d+)\.(weight|bias)", r"band_split.to_features.\1.1.\2"),
    (r"layers\.(\d+)\.(\d+)\.attn\.to_out\.weight", r"layers.\1.\2.layers.0.0.to_out.0.weight"),
    (r"layers\.(\d+)\.(\d+)\.attn\.(norm\.gamma|to_qkv\.weight|to_gates\.weight)",
     r"layers.\1.\2.layers.0.0.\3"),
    (r"layers\.(\d+)\.(\d+)\.ff\.norm\.gamma", r"layers.\1.\2.layers.0.1.net.0.gamma"),
    (r"layers\.(\d+)\.(\d+)\.ff\.net1\.(weight|bias)", r"layers.\1.\2.layers.0.1.net.1.\3"),
    (r"layers\.(\d+)\.(\d+)\.ff\.net4\.(weight|bias)", r"layers.\1.\2.layers.0.1.net.4.\3"),
    (r"masks\.(\d+)\.first\.(\d+)\.(weight|bias)", r"mask_estimators.\1.to_freqs.\2.0.0.\3"),
    (r"masks\.(\d+)\.second\.(\d+)\.(weight|bias)", r"mask_estimators.\1.to_freqs.\2.0.2.\3"),
    (r"final_norm\.gamma", r"final_norm.gamma"),
]
# Rotary tables the training code precomputes; the reference derives them from `freqs`.
DROPPED = {"cos_t", "sin_t", "cos_f", "sin_f"}

CONFIG = {
    "audio": {"chunk_size": 588800, "num_channels": 2, "sample_rate": 44100,
              "min_mean_abs": 0.0},
    "model": {
        "dim": DIM, "depth": DEPTH, "stereo": True, "num_stems": len(STEMS),
        "time_transformer_depth": 1, "freq_transformer_depth": 1,
        "linear_transformer_depth": 0, "freqs_per_bands": FREQS_PER_BANDS,
        "dim_head": DIM_HEAD, "heads": HEADS, "attn_dropout": 0.0, "ff_dropout": 0.0,
        "flash_attn": True, "dim_freqs_in": 1025, "stft_n_fft": 2048,
        "stft_hop_length": 512, "stft_win_length": 2048, "stft_normalized": False,
        "zero_dc": False, "mask_estimator_depth": 2, "mlp_expansion_factor": 4,
        "use_pope": False,
    },
    "training": {"instruments": STEMS, "target_instrument": None},
    "inference": {"batch_size": 1, "num_overlap": 2},
}


def rename(name: str) -> str:
    for pattern, target in RENAMES:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, target, name)
    raise SystemExit(f"no rename for {name}")


def average(paths):
    """The `model` dicts of `paths`, averaged in float64."""
    states = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in paths]
    return {k: sum(s[k].double() for s in states).div(len(states)).to(v.dtype)
            for k, v in states[0].items()}


def convert(state, dtype):
    out = {rename(k): v.to(dtype) for k, v in state.items() if k not in DROPPED}
    # The reference has a bias on the gates and stores the rotary frequencies;
    # the training code has neither, so these are the values that leave it unchanged.
    freqs = 1.0 / 10000 ** (torch.arange(0, DIM_HEAD, 2, dtype=torch.float64) / DIM_HEAD)
    for layer in range(DEPTH):
        for axis in range(2):
            attn = f"layers.{layer}.{axis}.layers.0.0"
            out[f"{attn}.to_gates.bias"] = torch.zeros(HEADS, dtype=dtype)
            out[f"{attn}.rotary_embed.freqs"] = freqs.float()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="the .ckpt to write; "
                    "the config goes beside it as .yaml")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    state = convert(average(args.ckpt), getattr(torch, args.dtype))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    config = args.out.with_suffix(".yaml")
    config.write_text(yaml.dump(CONFIG, sort_keys=False))
    print(f"{len(args.ckpt)} checkpoint(s) -> {args.out} ({len(state)} tensors), {config}")


if __name__ == "__main__":
    sys.exit(main())
