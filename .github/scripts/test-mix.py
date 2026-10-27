"""Write a 10 s stereo test mix at 44.1 kHz: a chord, a kick and some noise.

Usage: test-mix.py OUT.wav
"""

import math
import random
import struct
import sys
import wave

RATE = 44100
SECONDS = 10

random.seed(1)
frames = bytearray()
for i in range(RATE * SECONDS):
    t = i / RATE
    chord = 0.2 * sum(math.sin(2 * math.pi * f * t) for f in (220.0, 277.2, 329.6)) / 3
    beat = t % 0.5
    kick = 0.5 * math.sin(2 * math.pi * 60 * beat) * math.exp(-20 * beat)
    left = chord + kick + 0.02 * random.uniform(-1, 1)
    right = chord + kick + 0.02 * random.uniform(-1, 1)
    frames += struct.pack("<hh", *(int(max(-1.0, min(1.0, v)) * 32767) for v in (left, right)))

with wave.open(sys.argv[1], "wb") as out:
    out.setnchannels(2)
    out.setsampwidth(2)
    out.setframerate(RATE)
    out.writeframes(bytes(frames))
