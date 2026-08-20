"""Convert a BS-RoFormer checkpoint to safetensors for stemd-mlx.

Published RoFormer weights are PyTorch `.ckpt` state dicts and stemd runs MLX,
so a model has to be converted once before it can be loaded. This is the only
thing `pyproject.toml`'s torch dependency is for; nothing here runs at serve
time.

    uv run tools/export/convert_roformer.py \\
        --ckpt model_bs_roformer_ep_317_sdr_12.9755.ckpt \\
        --out  models/bs_roformer_viperx.safetensors

**Names are kept verbatim and nothing is transposed.** Both are deliberate.
The Rust side addresses tensors by name through `weights::Scope`, so a rename
here would be a rename there for no gain, and it would break the one property
that makes a conversion checkable: that the file still says what the original
said. Layout is the loader's business — `Linear` transposes its own weight
because the rearrangement belongs to the weight, and unlike demucs there is
nothing else to rearrange, since BS-RoFormer has no convolutions.

What this does check is that the tensors are the ones the architecture wants.
A checkpoint that loads `strict=True` into the reference implementation is one
whose names and shapes are all accounted for, which is a stronger statement
than "the file parsed" and costs one model construction.
"""

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def check_against_reference(state, config):
    """Construct the reference model and load `state` into it strictly.

    Skipped when the reference implementation is not importable, because it is
    not on PyPI in a form that loads these weights — the published package grew
    hyper-connections and no longer matches. See tools/eval/roformer_hybrid.py.
    """
    try:
        from models.bs_roformer.bs_roformer import BSRoformer
    except ImportError:
        print("  reference implementation not on PYTHONPATH; skipping the load check")
        return None

    spec = yaml.load(config.read_text(), Loader=yaml.UnsafeLoader)
    accepted = set(inspect.signature(BSRoformer.__init__).parameters)
    dropped = [k for k in spec["model"] if k not in accepted]
    if dropped:
        raise SystemExit(
            f"the reference implementation does not accept {dropped} — it is a "
            "different version from the one these weights were trained with"
        )
    model = BSRoformer(**{k: v for k, v in spec["model"].items() if k in accepted})
    model.load_state_dict(state, strict=True)
    print(f"  loads strictly into the reference: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="the matching .yaml, to check against")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"{args.ckpt} is a {type(state).__name__}, not a state dict")
    state = state.get("state_dict", state)
    print(f"{args.ckpt.name}: {len(state)} tensors")

    dtypes = {str(v.dtype) for v in state.values()}
    if dtypes != {"torch.float32"}:
        print(f"  note: dtypes are {sorted(dtypes)}, not float32 alone")

    if args.config:
        check_against_reference(state, args.config)

    # Contiguous because safetensors stores a flat buffer and a view would be
    # written in the wrong order without complaint. Cloned because it refuses
    # aliased tensors outright, and BS-RoFormer has plenty: every layer's
    # `rotary_embed.freqs` is one shared buffer, so twelve names point at the
    # same 32 floats. Duplicating them costs nothing and keeps the file a
    # faithful copy of what the checkpoint said, which is what lets the port
    # check its own rotary embedding against the original rather than
    # re-deriving one and hoping the constants match.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in state.items()}, str(args.out))

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    print(f"  sha256 {digest}")


if __name__ == "__main__":
    sys.exit(main())
