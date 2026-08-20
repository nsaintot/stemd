"""Time BS PolarFormer against the viperx BS-RoFormer, in one runtime.

The two score the same — 11.04 against 11.12, inside the noise floor — so the
only question left is what they cost. `tools/eval/polarformer.py` answers that
in multiply-accumulates, which is exact and ignores everything real: kernel
launches, memory traffic, and how well a narrower matmul keeps the GPU busy.

This puts a stopwatch on it instead. Weights are random, because speed does not
depend on their values — which means this needs no checkpoint for either model
and can time the viperx configuration without a 600 MB download.

Both are timed on the same device, with warmup, over the same duration of audio
rather than the same chunk count: they use 8.0 s and 13.35 s chunks, so
per-chunk timings are not comparable and per-second-of-audio ones are.

torch/MPS is not the runtime stemd ships, so the number that transfers is the
*ratio*. It captures what the arithmetic cannot — the fixed per-launch costs
that do not shrink with width — and both models meet them equally.

```text
uv run --extra eval tools/eval/roformer_speed.py --msst <checkout> \\
    --a <checkout>/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml \\
    --b model_bs_polarformer_float16.yaml
```
"""

import argparse
import inspect
import sys
import time

import torch
import yaml


def build(config, device, half):
    from models.bs_roformer.bs_roformer import BSRoformer

    cfg = yaml.load(open(config), Loader=yaml.UnsafeLoader)
    accepted = set(inspect.signature(BSRoformer.__init__).parameters)
    model = BSRoformer(**{k: v for k, v in cfg["model"].items() if k in accepted})
    model.eval().to(device)
    if half:
        model.half()
    chunk = int(cfg["audio"]["chunk_size"])
    params = sum(p.numel() for p in model.parameters())
    return model, chunk, cfg, params


def time_one(model, chunk, device, half, repeats):
    """Seconds per second of audio, best of `repeats` after a warmup."""
    dtype = torch.float16 if half else torch.float32
    x = torch.randn(1, 2, chunk, device=device, dtype=dtype)

    with torch.no_grad():
        model(x)  # warmup: the first call pays for shader compilation
    if device.type == "mps":
        torch.mps.synchronize()

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, best / (chunk / 44100)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msst", required=True)
    ap.add_argument("--a", required=True, help="config of the first model")
    ap.add_argument("--b", required=True, help="config of the second")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="float32 instead of float16")
    args = ap.parse_args()

    sys.path.insert(0, args.msst)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    half = not args.full
    print(f"{device}, {'float16' if half else 'float32'}, best of {args.repeats}\n")

    results = []
    for label, config in (("A", args.a), ("B", args.b)):
        model, chunk, cfg, params = build(config, device, half)
        per_chunk, per_second = time_one(model, chunk, device, half, args.repeats)
        m = cfg["model"]
        name = config.split("/")[-1].replace(".yaml", "")
        print(f"  {label}: {name}")
        print(
            f"     dim {m['dim']}  depth {m['depth']}  hop {m['stft_hop_length']}  "
            f"chunk {chunk / 44100:.2f}s  {params / 1e6:.1f}M params"
        )
        print(
            f"     {per_chunk * 1000:.0f} ms per chunk   {per_second:.4f} s per second "
            f"of audio   ({1 / per_second:.1f}x realtime)"
        )
        results.append((name, per_second))
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    (na, a), (nb, b) = results
    faster, slower = (nb, na) if b < a else (na, nb)
    print(f"\n  {faster} is {max(a, b) / min(a, b):.2f}x faster than {slower}")
    print("  (torch/MPS; what transfers to another runtime is this ratio, not the times)")


if __name__ == "__main__":
    main()
