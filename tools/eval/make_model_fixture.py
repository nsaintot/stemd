"""One whole segment through htdemucs, input and output.

The assembly test. Every component below this has its own passing null, so a
failure here is in the wiring — the interleaved loops, the skip order, the
normalisation, or the mask — rather than in the arithmetic.
"""
from pathlib import Path
import mlx.core as mx
import numpy as np

OUT = Path("fixtures"); OUT.mkdir(exist_ok=True)

def write(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    (OUT / f"{name}.shape").write_text(" ".join(str(d) for d in arr.shape))
    (OUT / f"{name}.f32").write_bytes(arr.tobytes())
    print(f"  {name}: {arr.shape}")

from demucs_mlx import Separator
sep = Separator(model="htdemucs", shifts=1, overlap=0.25, split=True,
                segment=None, batch_size=1, progress=False)
m = sep._model.models[0]

rng = np.random.default_rng(2718)
n = 44100  # a second: shorter than the 7.8 s segment, so the padding path runs
t = np.arange(n) / 44100.0
sig = (0.3 * np.sin(2 * np.pi * 196.0 * t)
       + 0.2 * np.sin(2 * np.pi * 1470.0 * t + 0.7)
       + 0.15 * np.exp(-40.0 * ((np.arange(n) % 22050) / 22050)) * rng.standard_normal(n))
audio = np.stack([sig, np.roll(sig, 311) * 0.85]).astype(np.float32)[None]
write("model_input", audio)

out = m(mx.array(audio))
mx.eval(out)
print("  output:", out.shape)
write("model_output", np.asarray(out))
