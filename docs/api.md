# API

```text
GET    /v1/health                     runtime, model, queue and cache state
POST   /v1/jobs                       raw PCM body -> 202 + job
                                      200 + a finished job on a cache hit
                                      429 when the queue is full
GET    /v1/jobs/{id}                  progress, then the result object
GET    /v1/jobs/{id}/stems/{name}     raw interleaved PCM stream
                                      410 if the stems were reaped: resubmit
DELETE /v1/jobs/{id}                  stop the job, drop the handle
GET    /v1/logs                       recent log lines
```

## Identifiers

Two ids cross the wire, and a client stores both. Neither ever contains anything
outside `[A-Za-z0-9._-]`, neither exceeds **31 characters**: an id plus its NUL
fits a 32-byte buffer, and neither is empty, a path separator, `.` or `..`. So
an id is safe to use directly as a path component, and a bounded copy is all the
filtering a client needs. The server owns that invariant against arbitrary
operator input, which is what `--demucs-model` is.

| Field | Shape | Example |
| --- | --- | --- |
| `model_id` (in `/v1/health`) | `<artefact>-<8 hex>` | `htdemucs-22c7f2c9` |
| job `id` (in `/v1/jobs`) | `<12 hex of the cache key>-<attempt>` | `a8e29ff15c16-3` |

`model_id` identifies *what produced the stems*: the artefact name, then 8 hex
of a digest over everything that decides the samples: the weights, how the
preset uses them, the precision the model runs at, and the segment overlap. Key
a client-side stem cache on it rather than on `model`, since several artefacts
can share one model name and `model` cannot tell you whether stems on disk came
from the weights now loaded.

All of it is in there deliberately, and the weights are the least interesting
part. Repointing a preset at a different artefact changes the id, which is
obvious; so does changing the arrangement of a preset whose weights are
untouched, which is not. `Balanced` did exactly that, it used to run all four of
`htdemucs_ft`'s models and ship `bass + other` as the harmonics, and now runs two
and ships the remainder. Same file, different audio. An id that covered only the
file would have let stems from before that survive as though they were current.
`--full-precision` and `--overlap` are there for the same reason: both change the
samples without touching a byte of the artefact.

**So the same artefact and preset do not give the same id on every machine.**
Precision is chosen per model against the backend, so `htdemucs-389aad7d` above
is what a GPU server reports; one with no usable GPU runs at f32 and reports
`htdemucs-01db1f92`, and a CUDA server on a chained preset reports an id derived
from both models' precisions. This is the point rather than a wrinkle: those
servers produce different samples, and an id that hid the difference would let a
client serve one machine's stems as though they were another's. Read the id, do
not predict it.

The server keys its own cache on the same fields, at full length, which you never
see. Treat `model_id` as opaque: the field list grows when something new turns
out to change the audio, and any id you have stored simply stops matching, which
is the outcome you want.

A job `id` identifies one *attempt* at one track. Its leading half is the cache
key, which is also what the server logs, so a log line and a URL can be matched
by eye. The trailing attempt number is what keeps two clients asking for the same
track at different times from sharing an id: without it, a `DELETE` from a
client that already finished would cancel the other's separation.

An artefact that is not a built-in preset (`--demucs-model` naming a
`.safetensors` you put in `--models` yourself) publishes `custom-<name>-<8 hex>`,
with the name sanitised and shortened to fit and the digest taken over the full
name.

## What a client must do

Four things. The reference implementation is
[stemd-cli](../crates/stemd-cli/src/main.rs).

### 1. Upload raw PCM

Not an encoded file. The client already has decoded samples, and decoding them
again server-side risks the two decoders disagreeing on encoder delay, which
shows up as a shallow null.

```
POST /v1/jobs?sample_rate=44100&channels=2&format=f32le
```

| parameter | default | |
|---|---|---|
| `sample_rate` | `44100` | rate of the uploaded body; converted to the model's if it differs |
| `channels` | `2` | |
| `format` | `s16le` | encoding of the uploaded body: `s16le` or `f32le` |
| `output_format` | `default_output_format` | `flac`, `wav`, `mp3`, `pcm16`, `pcm32` |
| `output_sample_rate` | `default_output_sample_rate` | `24000`, `44100`, `48000` or `96000` |
| `include_derived` | `false` | ship the derived part too, rather than leaving the client to rebuild it |
| `dsp_mode` | `0` | filter the output conversion runs through; see below |

The two defaults are set in the server's window and persist across its launches,
so read them from `/v1/health` rather than assuming, they are not constants.
Every parameter here is part of the cache key, so two jobs differing in any of
them are two separations.

Upload at the model's own rate where you can. Anything else is converted on
arrival, which costs time and puts a filter between the client's samples and the
model's. The body limit is the byte count of the longest accepted track at
96 kHz f32le, so a longer track at a higher rate is refused by length before the
duration gate sees it.

```bash
curl -X POST --data-binary @track.pcm \
  "http://host:8420/v1/jobs?sample_rate=44100&channels=2&format=s16le"
```

#### `dsp_mode`

Mode `0` is a general resampler and handles any rate pair. It is the default and
almost certainly what you want.

A numbered mode is a copy of one particular client's own converter, and exists
for one case: a client that rebuilds `drums = mix - harmonics - vocals` has to
resample its mix to the stems' rate, and that subtraction only cancels if both
sides went through the same filter. Two good resamplers are not enough. Against
ffmpeg's `swresample` ours agrees to about -77 dB of the mix, which lands as
-36 dB of a part sitting 41 dB below it; matching the filter takes that to
-101 dB.

Each numbered mode covers exactly one rate pair, listed in `dsp_mode_pairs` from
`/v1/health`, and is a `400` on any other. Nothing falls back: a client asking
for a filter it can match is never handed a different one under the same name.

A mode's delay is part of the filter. Mode 1's output sits 64 samples later at
96 kHz than mode 0's, which puts a feature exactly where the ratio does. That is
the copied converter's own group delay, and carrying it is the point: the mix the
client subtracts from went through the same taps.

```bash
# mode 1 covers 44100 -> 96000, so it needs that output rate.
curl -X POST --data-binary @track.pcm \
  "http://host:8420/v1/jobs?format=s16le&output_sample_rate=96000&dsp_mode=1"
```

### 2. Poll the returned job id

Treat it as the job to poll, not as proof a new one was created: a request
matching a separation already running returns *that* job. A `200` instead of
`202` means the answer was cached and is already complete.

Progress is a work counter, not a spinner:

```json
{"stage": "separating", "completed": 8, "total": 12, "fraction": 0.63}
```

Stages are `queued`, `analysing`, `separating`, `reconstructing`, `writing`,
`done`, `failed`, `cancelled`. `fraction` is weighted so separation owns
0.10–0.90. While `queued`, `completed`/`total` are the live queue position rather
than a chunk count.

`cancelled` is terminal and carries no `error`: it is what the client asked for.
In practice a client rarely sees it, since the handle it cancelled is gone by
then.

### 3. Download both stems and apply `gain`

```json
{
  "sample_rate": 44100, "channels": 2, "frames": 1323000,
  "format": "flac",
  "stems": [
    {"name": "harmonics", "path": "~/Library/Caches/stemd/<key>/harmonics.flac",
     "url": "/v1/jobs/{id}/stems/harmonics", "bytes": 2216440, "gain": 1.0}
  ],
  "model_residual_db": -35.2,
  "separation_secs": 1.39, "realtime_factor": 21.6,
  "cached": false
}
```

Multiply each stem by `1.0 / gain`. In any 16-bit format a stem peaking past full
scale is scaled to fit, so `gain` is not always `1.0`; ignoring it plays those
stems quiet. Only `pcm32` has the headroom to never need it.

The gain is per stem, not shared: quantising a quiet stem against the loudest
stem's peak throws away bits for nothing.

`separation_secs` is what the run that produced the stems cost, which for a
cached answer was an earlier job. `path` is useful only to a client on the same
filesystem; everything else should fetch `url`.

`model_residual_db` is diagnostic, not a gate: it is `mix − sum(model sources)`,
what the model failed to account for. **What it means depends on the preset, so
do not compare it across them.**

On `Fast` it is around −30 dB and it lands on the rebuilt `drums`, making it the
quality signal for that one control and nothing else. On `Balanced` and
`Quality` it reads about −160 dB, and that is not a better model, those presets
build `harmonics` as `mix − vocals − drums`, so the three parts are constructed
to sum and there is no redundancy left to measure. There the number means "the
arithmetic is right", and the leftover it used to describe is inside
`harmonics`.

The three parts sum exactly regardless of its value, on every preset.

### 4. Rebuild the third part while mixing

Never materialise it. Expanding `out = d·D + h·H + v·V` with `D = mix − H − V`
collapses to:

```
out = d·mix + (h−d)·H + (v−d)·V
```

Three multiplies and two adds per sample, against buffers the client already
holds: the same cost three transferred stems would have had. At
`d == h == v == 1.0` both stem coefficients are exactly zero and the output is
the untouched mix, **bit for bit**.

> Snap the controls to a literal `1.0`. Bit-exactness at unity depends on `(h−d)`
> being precisely zero, and 0.9997 does not qualify.

Reconstruction is exact at any *lossless* wire format, because the rebuilt part
comes from the *same* mix the parts are summed back against, so each stem's
quantisation error cancels. What quantisation costs is that part's purity, not
the sum. `mp3` is the one format this does not hold for: see below.

## Wire format

Stems ship as **FLAC** by default, carrying 16-bit samples. Four of the five
formats carry *exactly* the integers `pcm16` would have: FLAC, WAV and `pcm16`
are three containers over one set of samples, so which you ask for changes the
bytes on the wire and not the audio, and a client rebuilding the third part gets
the same answer from any of them.

| `output_format` | per five-minute stem | content type | |
|---|---|---|---|
| `flac` *(default)* | ~20–30 MB, content-dependent | `audio/flac` | lossless |
| `wav` | 53 MB | `audio/wav` | lossless |
| `pcm16` | 53 MB | `application/octet-stream` | lossless, no container |
| `pcm32` | 106 MB | `application/octet-stream` | lossless, no ceiling |
| `mp3` | ~12 MB at 320 kbps | `audio/mpeg` | **lossy** |

`s16le` and `f32le` are accepted as spellings of `pcm16` and `pcm32`.

FLAC lands around 40% of the raw size on real material: a sparse stem such as
vocals on an instrumental track compresses to a quarter. `pcm16` costs nothing to
read and is the right choice if the client cannot spare a decode; `wav` is the
same samples for anything that would rather be handed a header; `pcm32` exists
for an exact reconstruction null with no quantisation anywhere.

**`mp3` is not for a deck.** A perceptual codec hides its noise under a masking
threshold computed for the signal as encoded, and a stem exists to have its gain
changed afterwards: pull one down against the others and the noise that was
masked stops being masked. Lossy parts also do not sum, so the rebuilt third part
is no longer exact. It is the right choice for stems someone imports, auditions
or archives, and the wrong one for anything that moves the faders.

MP3 also constrains the rate: its modes stop at 48 kHz, so
`output_format=mp3&output_sample_rate=96000` is a `400` rather than a slow encode
to 48 kHz that no field in the response would have reported. `/v1/health`
lists the rates, and the window offers only the ones the chosen format carries.

Streams are **fixed block size**: STREAMINFO declares `min_blocksize ==
max_blocksize`, matching frames written with the blocking-strategy bit clear.
A stream that declares them unequal is telling the decoder to locate frames by
sample number instead of frame number, and a decoder that follows that either
stops partway through the file or reports every frame as out of order. The
encoder therefore excludes the trailing partial block from the declared minimum,
as libFLAC and ffmpeg do.

## Caching on the client

Key stems on `model_id`, which is safe to use directly as a directory name and
changes when the weights do: see [Identifiers](#identifiers) for why not
`model`.

Store the **two transferred stems only** and rebuild the third on playback, for
the same reason the wire carries two: it is a third less disk, and the sum stays
exact against whatever mix the client decodes.

Store them **losslessly**. Perceptual codecs place their noise under a masking
threshold computed for the signal as encoded, and a stem player exists to change
the relative gains, which is exactly what un-masks it. Solo a lossily-stored stem
and the coding noise is no longer hidden behind the other stems.

Stems arrive as FLAC, so the cheapest correct thing is to write those bytes
straight to disk: the encode already happened on the server, and storing the
stream verbatim costs no transcode in either direction. Ask for `pcm16` instead
if the device would rather spend disk than a decode. Not `mp3`, for the reason
under [Wire format](#wire-format): it is the format whose failure mode is
precisely a stem player.

## Discovery

The server advertises `_stemd._tcp`. TXT records carry enough to decide
compatibility without a round trip:

```
model=htdemucs  path=/v1  v=1  version=0.1.0
```

**`v=1` is the whole contract.** It means two stems (`harmonics` and `vocals`)
and a third the client rebuilds from the mix itself. That pair has never varied
and cannot: it is what protocol 1 *is*. If it ever changes, that is `v=2`, and a
client checking the version rejects the server before it has to wonder what it
was sent.

There used to be a `stems=harmonics,vocals` entry saying the same thing a second
time. It was dropped because a field that cannot vary invites a branch that can
never be taken: a client reading it would be writing code for a case protocol 1
does not have.

**`model` is a label, not an identity.** It names the artefact and says nothing
about how the preset uses it, so two servers can advertise the same `model` and
produce different audio: `Quality` did exactly that when it started chaining
its halves. Discovery is for finding a server and deciding you can talk to it.
Cache on `model_id` from `/v1/health`, which covers both, and which the section
above is about.

## Reference client

`stemd-cli` finds the server over mDNS (or takes `--host`), decodes wav, flac,
mp3, m4a, aac, aiff, mp4 and ogg via symphonia, uploads, shows progress, rebuilds
the third part exactly as a client should, and writes all three so they can be
auditioned without a DAW. It does not link the model at all: decoding lives in
`stemd-audio` precisely so the client stays a small binary.

```bash
cargo run --release -p stemd-cli -- track.mp3
cargo run --release -p stemd-cli -- track.wav --host 10.10.50.245:8420 --out ./stems
```

```
transfers    : 2 stems [harmonics, vocals]: `drums` is rebuilt here, never sent
separated    : 20.99s server-side (18.9x realtime)
  downloaded   : harmonics.flac  -15.2 dBFS
  downloaded   : vocals.flac     -22.7 dBFS
  rebuilt here : drums.flac      -10.8 dBFS
reconstruct  : -inf dB  (harmonics+vocals+drums vs the original)
model resid  : -35.2 dB (the model's own error, it all lands on `drums`)
```

`reconstruct` is the check that matters: it sums the three parts and nulls them
against the decode. Anything above roughly −120 dB means a part went missing, not
that the model separated badly.

All three files are written as FLAC at one shared scale, so they can be auditioned
against each other. `--wav-f32` writes unscaled float wavs for a DAW instead;
`--f32` uploads and receives float; `--dsp-mode` asks for a numbered filter and
is refused locally if this server does not offer it.

A file at any other rate is uploaded as it is and converted server-side, with a
note saying so. `mix - stems` needs both sides at one rate to line up, so `drums`
is rebuilt only when the stems come back at the file's own rate;
`--include-derived` has the server send it otherwise. Even when the two agree,
the stems have been through the model's rate and back while the mix on disk has
not, so `reconstruct` reads shallower than it does at 44.1 kHz throughout.
