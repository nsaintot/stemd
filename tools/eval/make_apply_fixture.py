"""A track longer than one segment, through the full apply path.

Long enough to need three overlapping segments, so the crossfade and the
weight-sum division are exercised rather than skipped.
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
# shifts=0, not 1: `shifts=1` is one *random-shift* pass, which bakes a
# random offset into the output. stemd wants determinism, and evaluation.md
# measured shifts as a net negative anyway.
sep = Separator(model="htdemucs", shifts=0, overlap=0.25, split=True,
                segment=None, batch_size=1, progress=False)

rng = np.random.default_rng(1123)
n = 44100 * 15   # 15 s: segment is 7.8 s, stride 5.85 s -> 3 segments
t = np.arange(n) / 44100.0
sig = (0.3 * np.sin(2 * np.pi * 174.0 * t)
       + 0.2 * np.sin(2 * np.pi * 880.0 * t + 0.3)
       + 0.15 * np.exp(-30.0 * ((np.arange(n) % 22050) / 22050)) * rng.standard_normal(n))
audio = np.stack([sig, np.roll(sig, 733) * 0.9]).astype(np.float32)
write("apply_input", audio)

_, stems = sep.separate_tensor(audio)
order = ["drums", "bass", "other", "vocals"]
out = np.stack([np.asarray(stems[k]) for k in order])
print("  sources:", order, "->", out.shape)
write("apply_output", out[None])   # [1, S, C, T]
