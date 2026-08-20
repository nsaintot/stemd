"""Dump a decoder step from each branch, with its skip connection.

The decoder returns two things — the layer output and the value from before the
transposed convolution, which the time branch injects into — and both matter, so
both are dumped.
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
rng = np.random.default_rng(909)

# Frequency decoder 0: chin 384 -> chout 192, over [B, C, F, T].
x = (rng.standard_normal((1, 384, 8, 8)) * 0.4).astype(np.float32)
skip = (rng.standard_normal((1, 384, 8, 8)) * 0.4).astype(np.float32)
write("dec0_input", x); write("dec0_skip", skip)
z, pre = m.decoder[0](mx.array(x), mx.array(skip), 8)
mx.eval(z, pre)
write("dec0_output", np.asarray(z)); write("dec0_pre", np.asarray(pre))

# Time decoder 0: chin 384 -> chout 192, over [B, C, T].
xt = (rng.standard_normal((1, 384, 32)) * 0.4).astype(np.float32)
skipt = (rng.standard_normal((1, 384, 32)) * 0.4).astype(np.float32)
write("tdec0_input", xt); write("tdec0_skip", skipt)
zt, pret = m.tdecoder[0](mx.array(xt), mx.array(skipt), 128)
mx.eval(zt, pret)
write("tdec0_output", np.asarray(zt)); write("tdec0_pre", np.asarray(pret))
