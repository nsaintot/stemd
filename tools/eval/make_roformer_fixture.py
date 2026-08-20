"""Dump BS-RoFormer's intermediates, so the Rust port can be nulled stage by stage.

The htdemucs port was built this way and it is why it works: every layer is a
place a port can differ, and a difference that is not caught at the layer it
happened in is a difference hunted for across the whole model. Each stage here
becomes one test in `crates/stemd-mlx/tests/roformer.rs`.

    PYTHONPATH=<ZFTurbo checkout> uv run tools/eval/make_roformer_fixture.py \\
        --ckpt ...ckpt --config ...yaml --out <fixture dir>

Needs the same three things `tools/eval/roformer_hybrid.py` does; see its
docstring. Writes about 40 MB at the default length, which is why it takes an
output directory rather than writing into the repo — the tests read it through
`STEMD_MLX_ROFORMER_FIXTURES`.

The input is one second rather than the model's 8-second chunk. BS-RoFormer
attends over time with a rotary embedding and has no learned position table, so
it takes any length, and a shorter one keeps the intermediates to a size worth
keeping on disk. The whole-chunk case is covered by the end-to-end null against
the reference on real audio, not here.
"""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from models.bs_roformer.bs_roformer import BSRoformer


def write(out: Path, name: str, tensor):
    """One `.f32` of little-endian floats and one `.shape` beside it."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4"))
    (out / f"{name}.f32").write_bytes(array.tobytes())
    (out / f"{name}.shape").write_text(" ".join(str(d) for d in array.shape))
    print(f"  {name:<22} {tuple(array.shape)}  {array.nbytes / 1e6:.1f} MB")


def deterministic_audio(channels: int, samples: int) -> torch.Tensor:
    """Noise from a fixed seed, so a rerun compares against the same fixture."""
    generator = torch.Generator().manual_seed(0x5EED)
    return torch.randn(channels, samples, generator=generator) * 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    spec = yaml.load(args.config.read_text(), Loader=yaml.UnsafeLoader)
    accepted = set(inspect.signature(BSRoformer.__init__).parameters)
    model = BSRoformer(**{k: v for k, v in spec["model"].items() if k in accepted})
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    rate = int(spec["audio"]["sample_rate"])
    audio = deterministic_audio(2, int(args.seconds * rate))

    # Hooks rather than a re-implementation of the forward: a re-implementation
    # is a second place for the port's mistakes to be made, and it would agree
    # with the port for exactly the wrong reason.
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    handles = [
        model.band_split.register_forward_hook(capture("bandsplit")),
        model.layers[0][0].register_forward_hook(capture("block0_time")),
        model.layers[0][1].register_forward_hook(capture("block0_freq")),
        model.final_norm.register_forward_hook(capture("final_norm")),
        model.mask_estimators[0].register_forward_hook(capture("mask")),
        # One attention and one feed-forward on their own, because a whole
        # transformer block agreeing tells you less than knowing which half did.
        model.layers[0][0].layers[0][0].register_forward_hook(capture("attn0")),
        model.layers[0][0].layers[0][1].register_forward_hook(capture("ff0")),
    ]

    with torch.no_grad():
        output = model(audio[None])

    for handle in handles:
        handle.remove()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"writing to {args.out}")
    write(args.out, "rof_input", audio)

    # The spectrogram, computed exactly as the model does, so the port's very
    # first stage has something to match before anything else is built on it.
    window = torch.hann_window(int(spec["model"]["stft_win_length"]))
    spectrum = torch.stft(
        audio,
        n_fft=int(spec["model"]["stft_n_fft"]),
        hop_length=int(spec["model"]["stft_hop_length"]),
        win_length=int(spec["model"]["stft_win_length"]),
        window=window,
        normalized=bool(spec["model"]["stft_normalized"]),
        return_complex=True,
    )
    write(args.out, "rof_stft_real", spectrum.real)
    write(args.out, "rof_stft_imag", spectrum.imag)

    for name in ("bandsplit", "attn0", "ff0", "block0_time", "block0_freq", "final_norm", "mask"):
        write(args.out, f"rof_{name}", captured[name])
    write(args.out, "rof_output", output)

    print(f"\n  input {audio.shape[1] / rate:.2f} s, output {tuple(output.shape)}")


if __name__ == "__main__":
    sys.exit(main())
