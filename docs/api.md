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
POST   /v1/sessions                   open a session: track length and output
PUT    /v1/sessions/{id}/pcm?offset=N PCM placed at track sample N
PUT    /v1/sessions/{id}/window       needle and margins, JSON
GET    /v1/sessions/{id}              what has arrived and what is ready
GET    /v1/sessions/{id}/stems/{name} one ready range, ?from=&to=
                                      409 if it is not ready
GET    /v1/sessions/{id}/pieces/{name}/{k}  piece k, encoded, as stored
DELETE /v1/sessions/{id}              end the session
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
| `parts` | `harmonics,vocals` | comma-separated parts to ship; see below |
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

#### `parts`

Every model separates four sources: `drums`, `bass`, `other` and `vocals`.
`harmonics` is `bass` and `other` together. A job lists the parts it wants:

| `parts` | ships | the client rebuilds |
|---|---|---|
| `harmonics,vocals` (default) | `harmonics`, `vocals` | `drums` |
| `bass,other,vocals` | `vocals`, `bass`, `other` | `drums` |
| `drums,vocals` | `vocals`, `drums` | `harmonics` |
| `drums,bass,other,vocals` | all four | nothing |

- Parts may not overlap: `harmonics,bass` is refused.
- They must leave out one part or none. The part left out is the one the client
  rebuilds as `mix` minus the stems, and the one `include_derived` adds. The
  result names it in `derived_stem`, which is `null` when nothing is left out.
- `parts` in `/v1/health` lists what the loaded model offers. Every preset offers
  all five.
- Stems ship in the order `harmonics`, `vocals`, `drums`, `bass`, `other`.
- `Balanced` runs a third model, `htdemucs_ft`'s bass specialist, when a job asks
  for `bass` or `other`: about 1.5x the time. `Fast` and `Quality` get every part
  from one pass.
- With nothing left out, `Fast` ships its model's four sources as they are, and
  they miss the mix by the model's residual (around −30 dB). `Balanced` and
  `Quality` build `other` as the remainder, so the four sum exactly.

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
  "model_residual_db": -35.2, "derived_stem": "drums",
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

## Sessions

A session separates one track in pieces, around a playhead, in the order the
client asks. Use it to get stems near the needle within a segment or two instead
of after the whole track.

Once a session's whole upload has arrived, the server hashes it into a
[track id](#track-id). A session whose audio matches a track another session
already has open joins it, so two decks playing one song share the model work
and the memory from then on. Finished pieces are written to disk under that id,
survive restarts, and serve any later session on the same audio without a model
run.

All positions are track samples at the model's rate (`sample_rate` in
`/v1/health`), whatever the output rate.

### 1. Open

```
POST /v1/sessions?frames=<track length>&format=f32le&output_format=f32le&mean=<m>&std=<s>&peak=<p>
```

| Parameter | |
| --- | --- |
| `frames` | Required. Track length. |
| `sample_rate`, `channels`, `format` | The upload. `sample_rate` must be the model's rate. |
| `output_format`, `output_sample_rate`, `parts`, `include_derived`, `dsp_mode` | As for `/v1/jobs`. `mp3` is refused. |
| `mean`, `std` | Optional. Mean and standard deviation of the mono mixdown `(L + R) / 2` over the whole track, in f64, sent as f32. |
| `peak` | Optional. Largest absolute sample of the mix, both channels. Sets the gain, and so the piece directory: send it to share pieces with whole-track jobs, which use the true peak. |

The body is optional PCM from sample 0. The answer is `201` and the session view.
`429` means the `--max-sessions` limit is reached; `409` means the loaded model cannot run
segment by segment.

### Track id

The server computes it, as it does a job's cache key: SHA-256 over

1. three fields, each a little-endian `u32` byte length then its UTF-8 bytes:
   `stemd-track-1`, the sample rate (`44100`), the channel count (`2`);
2. every frame's samples as little-endian `f32`, channel by channel within the
   frame, at the model's rate.

It is the first 24 lowercase hex characters, the same whether the upload was
`s16le` or `f32le`. The view's `track` holds it once the whole track has
arrived, with `verified: true`.

When the upload completes:

| The server finds | It does |
| --- | --- |
| an open track with the same id and parts | moves the session onto it; what that track separated is ready at once |
| pieces on disk for the id and this output | lists them in `pieces` and drops the segments they cover from the plan |
| neither | writes this track's pieces as they complete |

Until then a session separates on its own, cue region first as usual. A second
deck loading a song already open separates its cue once, then joins. If the
client sent `mean` and `std` and they are not the mix's, the track is never
shared or stored, since its stems differ from a whole-track run's.

### 2. Upload

```
PUT /v1/sessions/{id}/pcm?offset=<sample>
```

Any piece, in any order, in the session's `format`. Send the cue region first,
and the rest after: nothing is shared or stored until the whole track has
arrived.

A sample uploaded twice must match to the bit; a difference is `409`.

### 3. Ask for a window

```json
PUT /v1/sessions/{id}/window
{"needle": 661500,
 "ahead":  {"frames": 705600, "priority": "critical"},
 "behind": {"frames": 88200,  "priority": "bonus"}}
```

| Priority | Runs |
| --- | --- |
| `critical` | before whole-track jobs, also between two segments of one |
| `bonus` | when no job is waiting |

Each request replaces the last one's plan; segments already run are kept. The
segments under the needle go first, at the higher of the two priorities, then
both sides outward, two from the higher-priority side for each one from the
other. Only what a window asks for is separated.

### 4. Read ready ranges

`GET /v1/sessions/{id}` lists `arrived` and `ready` as `[from, to]` pairs. A
sample is ready once every segment covering it has run: the last ~25% of a
segment waits for its neighbour.

```
GET /v1/sessions/{id}/stems/{name}?from=<sample>&to=<sample>
```

| Header | |
| --- | --- |
| `x-stemd-from` | First sample of the body, at the output rate. |
| `x-stemd-frames` | Samples in the body. |
| `x-stemd-gain` | Multiply by `1.0 / gain`. |
| `x-stemd-clipped` | Samples clipped to fit a 16-bit format. |

`from..to` must sit inside one ready range. The derived part is served when
`include_derived` is set.

### Pieces

Piece `k` is track samples `[k * stride_frames, (k + 1) * stride_frames)`. The
view's `pieces` lists those stored for this session's output; fetch one whole,
already encoded, with

```
GET /v1/sessions/{id}/pieces/{name}/{k}
```

It carries the same `x-stemd-*` headers. A stored piece is sent as it is; one
that is ready in memory but not stored yet is encoded on the way out. A segment
whose pieces are all stored is not run again, so a cached window shows in
`pieces` rather than in `ready`.

On disk, under `--pieces-dir` (default `stemd-pieces` in the platform's cache
directory, capped by `--pieces-max-gb`, least recently used first):

```
<model_id>/<format>-<rate>[-dsp<N>]-<parts>-g<gain>/<track id>/<stem>/<k>.<ext>
htdemucs-22c7f2c9/flac-44100-harmonics.vocals-g1/f9c29cbfe3466b674f9903d9/vocals/0003.flac
```

The derived part is always stored beside the others. A whole-track job writes
its pieces here too, so a session on a track already separated by a job starts
with every piece.

- **Gain** is one value for the whole session, since a stem's own peak is only
  known once all of it is separated: 3 dB over `peak`, or over full scale when
  `peak` is not given. `f32le` always ships at 1.0.
- **At another output rate** a range is converted with up to 8192 samples of
  context on each side, taken from the ready range around it. Its edges can
  differ slightly from a whole-track job's.

### 5. End

`DELETE /v1/sessions/{id}`, or let it go idle: a session with nothing to run and
no request for `--session-idle-secs` (default 30) is ended. A model switch ends
every session.

| Segments | Fast, Balanced | Quality |
| --- | --- | --- |
| length | 343,980 samples (7.8 s) | 588,800 samples (13.35 s) |
| step at `--overlap 0.25` | 257,985 samples | 441,600 samples |

Read the exact values from the session view's `segment_frames`, `stride_frames`
and `segments`.

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

**`v=1` is the whole contract.** It means a job that names no `parts` gets two
stems (`harmonics` and `vocals`) and a third the client rebuilds from the mix
itself. `parts` changes that only for a client that asks. Changing the default
is protocol 2, `v=2`, which a client checking the version rejects.

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
