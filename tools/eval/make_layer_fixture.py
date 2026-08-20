"""Dump the first frequency encoder layer's input and output.

The riskiest single step in the port: convolution over the wrong axis has the
right output shape and no error, so only a null against the reference catches
it. The dilated branch and the gated rewrite ride along in the same tensor.
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
bag = sep._model
model = bag.models[0] if hasattr(bag, "models") else bag

rng = np.random.default_rng(4242)
enc0 = model.encoder[0]
print("encoder[0]:", type(enc0).__name__, "freq", enc0.freq, "empty", enc0.empty)

# The shape the first encoder sees: [B, C*2 (cac), F, T].
# Eight frames rather than forty-four: a convolution over the wrong axis
# is just as wrong in a short tensor, and fixtures are committed.
x = rng.standard_normal((1, 4, 2048, 8)).astype(np.float32) * 0.5
write("enc0_input", x)
y = enc0(mx.array(x), None)
mx.eval(y)
write("enc0_output", np.asarray(y))

# And the time branch's first encoder, a 1-D layer over [B, C, T].
tenc0 = model.tencoder[0]
print("tencoder[0]:", type(tenc0).__name__, "freq", tenc0.freq, "empty", tenc0.empty)
xt = rng.standard_normal((1, 2, 8192)).astype(np.float32) * 0.5
write("tenc0_input", xt)
yt = tenc0(mx.array(xt))
mx.eval(yt)
write("tenc0_output", np.asarray(yt))
