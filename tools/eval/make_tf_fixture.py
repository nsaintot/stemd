"""Dump the cross-domain transformer, and its position embeddings on their own.

The embeddings are pure functions of the shape, so checking them separately
turns "the transformer is 20 dB off" into "the interleaving is wrong", which is
the difference between a bug you find and one you stare at.
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

from demucs_mlx.mlx_transformer import create_sin_embedding, create_2d_sin_embedding
write("emb_1d", np.asarray(create_sin_embedding(13, 512)))
write("emb_2d", np.asarray(create_2d_sin_embedding(512, 6, 5)))

from demucs_mlx import Separator
sep = Separator(model="htdemucs", shifts=1, overlap=0.25, split=True,
                segment=None, batch_size=1, progress=False)
ct = sep._model.models[0].crosstransformer

rng = np.random.default_rng(31337)
# The shapes the transformer actually sees after the channel upsampler:
# [B, 512, Fr, T1] and [B, 512, T2], kept small.
x = (rng.standard_normal((1, 512, 6, 5)) * 0.3).astype(np.float32)
xt = (rng.standard_normal((1, 512, 20)) * 0.3).astype(np.float32)
write("tf_x_input", x); write("tf_xt_input", xt)
ox, oxt = ct(mx.array(x), mx.array(xt))
mx.eval(ox, oxt)
write("tf_x_output", np.asarray(ox)); write("tf_xt_output", np.asarray(oxt))
