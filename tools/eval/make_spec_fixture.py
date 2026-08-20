"""Dump the Python MLX spectrogram for a known input, as a fixture to null against.

Taken from the real loaded model rather than reconstructed from the STFT
parameters, so the fixture carries demucs's own padding and trimming — the
`_spec` in mlx_htdemucs.py, not a textbook STFT. Those two differ, and the
difference is exactly the kind of thing a port gets wrong silently.
"""

import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path("fixtures")
OUT.mkdir(exist_ok=True)


def write(name: str, arr: np.ndarray) -> None:
    """Raw f32 plus a shape line, so Rust needs no npy parser."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    (OUT / f"{name}.shape").write_text(" ".join(str(d) for d in arr.shape))
    (OUT / f"{name}.f32").write_bytes(arr.tobytes())
    print(f"  {name}: {arr.shape}")


def main() -> None:
    from demucs_mlx import Separator

    sep = Separator(model="htdemucs", shifts=1, overlap=0.25, split=True,
                    segment=None, batch_size=1, progress=False)
    bag = sep._model
    model = bag.models[0] if hasattr(bag, "models") else bag
    print("model:", type(model).__name__, "hop", model.hop_length)

    # Deterministic, and not a pure tone: a tone would hide a bin-ordering
    # mistake behind a spectrum that is zero almost everywhere.
    rng = np.random.default_rng(20260815)
    # Deliberately small and deliberately not a multiple of the hop: the
    # padding and trimming are what this checks, and they do not need a long
    # signal to be wrong. Fixtures live in the repository, so their size is a
    # cost paid by every clone forever.
    n = 12000
    t = np.arange(n) / 44100.0
    sig = (
        0.3 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 1310.0 * t + 0.4)
        + 0.05 * rng.standard_normal(n)
    )
    audio = np.stack([sig, np.roll(sig, 137) * 0.8]).astype(np.float32)
    x = mx.array(audio)[None]  # [B=1, C=2, T]
    write("spec_input", np.asarray(x))

    z = model._spec(x)
    mx.eval(z)
    print("  _spec ->", z.shape, z.dtype)
    write("spec_real", np.real(np.asarray(z)))
    write("spec_imag", np.imag(np.asarray(z)))

    mag = model._magnitude(z)
    mx.eval(mag)
    write("spec_magnitude", np.asarray(mag))

    # And the way back, which has its own padding and trimming.
    back = model._ispec(z, length=n)
    mx.eval(back)
    write("ispec_output", np.asarray(back))

    err = np.asarray(back) - audio[None]
    rms = float(np.sqrt(np.mean(err**2)))
    ref = float(np.sqrt(np.mean(audio**2)))
    print(f"  python round trip: {20 * np.log10(rms / ref):.1f} dB vs the input")

    (OUT / "params.txt").write_text(
        f"hop_length {model.hop_length}\nn_fft {model.nfft}\nlength {n}\n"
    )
    print("  params:", (OUT / "params.txt").read_text().replace("\n", " "))


if __name__ == "__main__":
    main()
