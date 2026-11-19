# API

The HTTP contract between stemd-server and its clients, protocol 2. Every route
is under `/v2`. Protocol 1, under `/v1`, is deprecated and may stop working in
a future release: see [Protocol 1 (deprecated)](#protocol-1-deprecated).

```text
GET    /v2/health                                  server, model and defaults
POST   /v2/jobs                                    separate a whole track
GET    /v2/jobs/{id}                               progress, then the result
GET    /v2/jobs/{id}/stems/{name}                  one stem of a finished job
DELETE /v2/jobs/{id}                               release or cancel the job
PUT    /v2/filters/{id}                            upload a conversion filter
GET    /v2/filters/{id}                            summary of a stored filter
POST   /v2/sessions                                open a session
PUT    /v2/sessions/{id}/pcm?offset=N              upload audio at track sample N
PUT    /v2/sessions/{id}/window                    the regions to separate
GET    /v2/sessions/{id}                           the session view; long-poll with ?wait=
GET    /v2/sessions/{id}/stems/{name}              a range, or the whole stem
GET    /v2/sessions/{id}/pieces/{name}/{k}         piece k, encoded
DELETE /v2/sessions/{id}                           end the session
GET    /v2/separations/{track id}                  what the server holds for one output
GET    /v2/separations?file={file key}             the same, found by the client's file key
GET    /v2/separations/{track id}/stems/{name}     one stem, stored whole
GET    /v2/separations/{track id}/bundle           every stem in one file
GET    /v2/logs                                    recent log lines
```

- **Jobs** separate a whole track and return finished stems.
- **Sessions** separate a track in pieces around a playhead, for a deck that
  plays before the separation ends.
- **Separations** tell a client what is already stored, by track id, before it
  uploads anything.

The reference client is [stemd-cli](../crates/stemd-cli/src/main.rs).

## Basics

### Errors

An error body is JSON:

```json
{"error": "no filter 3fa4…; upload it with PUT /v2/filters/3fa4…", "code": "unknown_filter"}
```

`code` is present only where a client acts on it:

| `code` | Status | The client |
| --- | --- | --- |
| `unknown_filter` | `404` | uploads the filter, then resends |
| `unknown_separation` | `404` | opens a session and uploads, or submits a job |
| `protocol_1` | `410` | uses `/v2/sessions`: sessions are on `/v2` only |

A malformed query or JSON body is a plain-text `400`, `415` or `422`; a
malformed window is a JSON `400`. Every
route, `/v2/health` included, answers `503` while the server is switched off in
its window.

### Health

`GET /v2/health`:

```json
{"version": "0.2.1", "backend": "mlx", "model": "htdemucs", "device": "gpu",
 "sample_rate": 44100, "channels": 2, "preset": "Fast",
 "model_id": "htdemucs-22c7f2c9",
 "completed_jobs": 0, "queue_depth": 0,
 "cached_tracks": 0, "cached_bytes": 0, "cache_max_bytes": 4000000000,
 "max_track_seconds": 600.0,
 "output_sample_rates": [24000, 44100, 48000, 96000],
 "dsp_modes": [0, 1], "dsp_mode_pairs": {"1": [44100, 96000]},
 "parts": ["harmonics", "vocals", "drums", "bass", "other"],
 "default_output": {"format": "flac", "sample_rate": 44100,
                    "parts": ["harmonics", "vocals"], "derived": "drums"},
 "features": [],
 "segment_run_seconds": 0.41,
 "session": {"methods": {"partial": {"requires": ["mean", "std"], "optional": ["peak"]},
                         "whole": {"requires": [], "optional": ["peak"]}}}}
```

| Field | |
| --- | --- |
| `sample_rate`, `channels` | The model's rate and channel count. Positions in sessions are samples at this rate. |
| `preset` | `Fast`, `Balanced` or `Quality`; absent for a custom model. |
| `model_id` | What produced the stems. Key a cache on it, see [Identifiers](#identifiers). |
| `running_job` | The job id running now; absent when idle. |
| `max_track_seconds` | Longest accepted track. Longer is `413`. |
| `output_sample_rates` | Values `output_sample_rate` accepts. |
| `dsp_modes`, `dsp_mode_pairs` | Values `dsp_mode` accepts, and the one rate pair each numbered mode covers. |
| `parts` | Parts a request can list in `parts`. |
| `default_output` | The output of a request that names no format, rate or parts. Set in the window; read it, do not assume it. |
| `features` | Behaviors added after protocol 2; empty for now. |
| `segment_run_seconds` | Median seconds the loaded model takes per segment, over the last 16; `null` until one ran, and again after a model switch. |
| `session` | [Upload methods](#upload-methods) of a session; `null` when the model cannot run segment by segment. |

Read `/v2/health` again after a model switch: `model_id`, `preset`,
`segment_run_seconds` and `session` change with the model.

### The output block

Job results, session views and lookups describe their stems with one shape:

```json
"output": {"format": "flac", "sample_rate": 44100, "gain": 0.7071068,
           "parts": ["harmonics", "vocals"], "derived": "drums"}
```

| Field | |
| --- | --- |
| `format` | `flac`, `wav`, `mp3`, `pcm16` or `pcm32`. |
| `sample_rate` | Output rate. |
| `gain` | Multiply every sample by `1.0 / gain`. Absent in job results, where each stem carries its own, and for an open track. |
| `parts` | The parts shipped. |
| `derived` | The part the client rebuilds as `mix` minus the parts, or `null` when nothing is left out. |

## Identifiers

| Id | Shape | Example |
| --- | --- | --- |
| `model_id` | `<artefact>-<8 hex>`, or `custom-<name>-<8 hex>` | `htdemucs-22c7f2c9` |
| job `id` | `<12 hex>-<attempt>` | `a8e29ff15c16-3` |
| session `id` | `s<n>-<8 hex>` | `s1-0a1b2c3d` |
| track id | 24 lowercase hex | `f9c29cbfe3466b674f9903d9` |
| file key | 24 lowercase hex, made by the client | `3b0c55e19a7d42f08c6e1d2a` |

Every id is at most 31 characters of `[A-Za-z0-9._-]`, never `.` or `..`, so it
is safe as a path component.

- **`model_id`** covers the weights and how the preset uses them. Two presets on
  one artefact, or one preset that changes its arrangement, have different ids.
  It does not cover precision or segment overlap. Treat it as opaque.
- **Job `id`**: the first half is the cache key, the same for every request
  with the same audio and output; the attempt number keeps two clients' jobs
  apart.

### Track id

SHA-256 over:

1. three fields, each a little-endian `u32` byte length then its UTF-8 bytes:
   `stemd-track-1`, the sample rate (`44100`), the channel count (`2`);
2. every sample as little-endian `f32`, interleaved, at the model's rate.

The id is the first 24 hex characters. An `s16le` sample `v` hashes as the `f32`
`v / 32768.0`, so `s16le` and `f32le` uploads of the same audio share an id.

### File key

The track id needs the whole file decoded. A file key finds a track without
decoding: the client derives it from the encoded file, for example from its size,
its decoded length and a sample of its bytes. The server never computes it; it
only stores and matches it.

- Send `file=<key>` on `POST /v2/sessions` or `POST /v2/jobs`. Once the server
  has hashed the whole upload, it records `key → track id`, persisted with the
  stored stems.
- `GET /v2/separations?file=<key>&<output>` then answers as a lookup by that
  track id, `track` included.
- A key that arrives with a second, different track id is ambiguous from then
  on: lookups by it answer `404 unknown_separation`, and clients upload.
- Clients that share a server share entries only when they compute the key the
  same way and their decoders give the same samples. A decoder that trims
  delay or padding differently gives a different track id.

### Track statistics

Sessions take three optional values computed over the whole decoded track, at
the model's rate. The server checks `mean` and `std` against its own to the bit;
a mismatch means the track is never shared or stored.

```text
m[i] = (f64(L[i]) + f64(R[i])) / 2
mean = (m[0] + m[1] + … + m[n-1]) / n                f64, summed in index order
var  = ((m[0] − mean)² + … + (m[n-1] − mean)²) / n   f64, in index order, divided by n
std  = max(sqrt(var), 1e-8)
peak = max |x| over every sample of both channels
```

Send `mean` and `std` as `f32`, written as the shortest decimal that parses back
to the same `f32`. Use a plain loop: a compensated or pairwise sum, such as
Python's built-in `sum` since 3.12, can round differently.

## Audio

### Upload

Raw interleaved PCM, not an encoded file:

| `format` | Sample |
| --- | --- |
| `s16le` (default) | `i16`, read as `v / 32768.0` |
| `f32le` | `f32` |

Upload at the model's `sample_rate`. A job converts any other rate on arrival;
a session refuses it.

### Output formats

| `output_format` | Samples | Content type | 5-minute stem |
| --- | --- | --- | --- |
| `flac` | 16-bit | `audio/flac` | 20 to 30 MB |
| `wav` | 16-bit | `audio/wav` | 53 MB |
| `pcm16` | 16-bit, raw interleaved | `application/octet-stream` | 53 MB |
| `pcm32` | `f32`, raw interleaved | `application/octet-stream` | 106 MB |
| `mp3` | lossy, 320 kbps | `audio/mpeg` | 12 MB |

- `flac`, `wav` and `pcm16` carry the same integers. `s16le`, `s16`, `f32le`
  and `f32` are accepted spellings of `pcm16` and `pcm32`; responses always say
  `pcm16` and `pcm32`.
- A 16-bit sample is `round(clamp(x × gain, −1, 1) × 32767)`. Restore it as
  `v / 32767 / gain`.
- `pcm32` has headroom: its gain is always `1.0`. In a 16-bit format a sample
  past full scale after the gain is clipped and counted in `x-stemd-clipped`.
- `mp3` is for stems someone imports or archives, not for a deck: moving a
  fader unmasks its coding noise, and lossy parts do not sum to the mix. It
  stops at 48 kHz (`96000` with `mp3` is `400`), and sessions refuse it.
- FLAC streams use one fixed block size.

### Parts

Every model separates `drums`, `bass`, `other` and `vocals`; `harmonics` is
`bass` and `other` together. A request lists the parts to ship in `parts`:

| `parts` | Ships | The client rebuilds |
| --- | --- | --- |
| `harmonics,vocals` (default) | `harmonics`, `vocals` | `drums` |
| `bass,other,vocals` | `vocals`, `bass`, `other` | `drums` |
| `drums,vocals` | `vocals`, `drums` | `harmonics` |
| `drums,bass,other,vocals` | all four | nothing |

- Parts may not overlap, and must leave out one part or none.
- `include_derived=true` ships the part left out as well.
- Stems come in the order `harmonics`, `vocals`, `drums`, `bass`, `other`, with
  the derived part last.
- `Balanced` runs a third model when `bass` or `other` is asked for: about 1.5x
  the time.

### Rebuilding the derived part

Do not store it. Mix it on the fly from the decoded mix and the stems:

```text
out = d·mix + (h − d)·H + (v − d)·V        h, v, d: the three faders
```

At `h = v = d = 1.0` the output is the mix, bit for bit. Snap faders to a
literal `1.0` for that. The sum is exact in every lossless format, since the
derived part comes from the same mix it is summed against.

## Jobs

### Submit

```
POST /v2/jobs?sample_rate=44100&channels=2&format=s16le     body: the whole track
```

| Parameter | Default | |
| --- | --- | --- |
| `sample_rate`, `channels`, `format` | `44100`, `2`, `s16le` | The upload. |
| `output_format` | `default_output.format` | See [Output formats](#output-formats). |
| `output_sample_rate` | `default_output.sample_rate` | One of `output_sample_rates`. |
| `parts` | `default_output.parts` | Comma-separated, see [Parts](#parts). |
| `include_derived` | `false` | `true` ships the derived part too. |
| `dsp_mode` | `0` | The output rate converter. `0` handles any pair; numbered modes are deprecated. |
| `dsp_filter` | none | Id of an [uploaded filter](#filters) to convert with instead. |
| `file` | none | The client's [file key](#file-key), recorded when the job's stems are stored. |

| Status | |
| --- | --- |
| `202` | Accepted; poll the job. A request matching a job already running returns that job. |
| `200` | Cached: the job is already done. |
| `400` | A bad parameter, named in `error`. |
| `404` | `unknown_filter`. |
| `413` | Longer than `max_track_seconds`, or a body over the upload limit. |
| `429` | The queue is full. |
| `507` | The disk is full. |

### Poll

```json
GET /v2/jobs/{id}
{"id": "a8e29ff15c16-1",
 "progress": {"stage": "done", "completed": 12, "total": 12, "fraction": 1.0, "detail": null},
 "result": {
   "output": {"format": "flac", "sample_rate": 44100,
              "parts": ["harmonics", "vocals"], "derived": "drums"},
   "channels": 2, "frames": 1323000,
   "stems": [{"name": "harmonics", "url": "/v2/jobs/a8e29ff15c16-1/stems/harmonics",
              "bytes": 2216440, "gain": 1.0},
             {"name": "vocals", "url": "/v2/jobs/a8e29ff15c16-1/stems/vocals",
              "bytes": 1120333, "gain": 1.0}],
   "model_residual_db": -35.2, "separation_secs": 1.39, "realtime_factor": 21.6,
   "cached": false}}
```

- `stage`: `queued`, `analysing`, `separating`, `reconstructing`, `writing`,
  then `done`, `failed` (with `error`) or `cancelled`.
- `fraction` runs 0 to 1; separation covers 0.10 to 0.90. While `queued`,
  `completed` and `total` are the queue position and `detail` says so (`next`,
  `2 ahead`).
- `result` is present once `done`. `frames` is at the output rate. Each stem has
  its own `gain`.
- `model_residual_db` is diagnostic: `mix − sum(model sources)`. Around −30 dB
  on `Fast`, where it lands in the rebuilt `drums`; about −160 dB on `Balanced`
  and `Quality`, which build one part as the remainder. Do not compare it
  across presets.
- `separation_secs` is the cost of the run that made the stems, an earlier one
  for a cached answer.

### Download and release

`GET /v2/jobs/{id}/stems/{name}` sends the stem in `output_format`.

| Status | |
| --- | --- |
| `200` | The stem. |
| `404` | No such job or stem. |
| `409` | The job has not finished. |
| `410` | The stems were evicted: submit again. |

`DELETE /v2/jobs/{id}` releases the job (`204`). The last client to release a
running job cancels it.

A finished job also stores its stems for [lookups](#separations) and sessions on
the same track.

### Filters

A client that rebuilds the derived part at another rate must resample its mix
through the same filter as the stems. It uploads its converter once and names
it in `dsp_filter`:

```
PUT /v2/filters/<id>     body: the filter file; 201 when new, 200 when stored already
GET /v2/filters/<id>     the summary, or 404 unknown_filter
```

`<id>` is the lowercase hex SHA-256 of the file. Both answers carry
`{"id", "in_rate", "out_rate", "taps", "branch_taps"}`. The file, little-endian:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `SFIR` |
| 4 | 4 | version `u32`, `1` |
| 8 | 4 | input rate `u32`: the model's `sample_rate` |
| 12 | 4 | output rate `u32`: one of `output_sample_rates`, not the input rate |
| 16 | 4 × n | prototype taps, `f32` |

The filter is a stateless polyphase FIR. With `L / M = out / in` in lowest terms
and `n = L × B` taps `p`, output frame `m` is
`y[m] = Σ x[k] · p[m·M − k·L]`, with `p` and `x` zero outside their range. The
output has `ceil(frames · L / M)` frames. Taps are applied as given (unity gain
means they sum to `L`), and no delay is compensated. `B` is at most 1024 and `n`
at most 1,048,576.

- A malformed file, or one whose hash is not `<id>`, is `400`; an oversized one
  is `413`.
- The server keeps 64 filters across restarts and evicts the least recently
  used. A job naming an evicted filter gets `404 unknown_filter`.
- A job with `dsp_filter` must ask for exactly the filter's rate pair, and may
  not also set a non-zero `dsp_mode`.

## Sessions

A session separates one track segment by segment, in the order its window asks.
Positions are samples at the model's rate.

```text
open ──▶ upload ──▶ window ──▶ long-poll the view ──▶ read ranges ──▶ end
```

Two sessions on the same audio share one track: the model work, the memory and
the stored pieces. Finished pieces are stored under the [track id](#track-id)
and serve any later session or lookup.

### Open

```
POST /v2/sessions?frames=13230000&format=f32le&output_format=flac
```

| Parameter | |
| --- | --- |
| `frames` | Track length. Required unless `track` is set. |
| `track` | A track id the lookup reports as `open`: join it and upload nothing. |
| `file` | Optional. The client's [file key](#file-key). |
| `sample_rate`, `channels`, `format` | The upload. Rate and channels must be the model's. |
| output parameters | As for jobs. `mp3` is refused. |
| `mean`, `std` | Optional, both or neither. See [Upload methods](#upload-methods) and [Track statistics](#track-statistics). |
| `peak` | Optional. Sets the session's gain, see [Gain](#gain). |

The body is optional audio from sample 0; it must be empty with `track`.

| Status | |
| --- | --- |
| `201` | The [session view](#view). |
| `400` | A bad parameter: wrong rate or channels, `frames=0`, only one of `mean` and `std`, `frames` that differs from the joined track. |
| `404` | `unknown_separation`: the track to join is no longer open. Open with `frames` and upload. |
| `409` | The loaded model cannot run segment by segment. |
| `413` | Longer than `max_track_seconds`. |
| `429` | The session limit is reached. |

### Upload methods

`session.methods` in `/v2/health` says what each way of uploading needs on the
loaded model:

| Method | The open body | Segments run before the upload ends when |
| --- | --- | --- |
| `whole` | the whole track | always; `requires` is empty |
| `partial` | a prefix, or nothing; the rest follows with `PUT /pcm` | the open sent every name in `requires` |

| Model | `partial` requires |
| --- | --- |
| `htdemucs`, `htdemucs_ft` (Fast, Balanced) | `mean`, `std` |
| `stemd_roformer` (Quality) | nothing |

Without the required values a `partial` session still works, but no segment runs
until the last sample arrives. Each name in `requires` and `optional` is an open
parameter; a changed definition gets a new name. A client:

1. ignores keys it does not know;
2. uses a method only when it knows every name in its `requires`, and otherwise `whole`;
3. ignores methods it does not know;
4. leaves out an optional parameter it cannot compute exactly.

### Gain

One gain for the whole session, fixed at open: 3 dB over `peak`, or over full
scale without it.

```text
gain = 1 / (peak × √2)   when that is below 1, else 1      (pcm32: always 1)
```

Without `peak` a 16-bit session ships at `0.7071068`. A stem that still passes
full scale is clipped, and `x-stemd-clipped` counts the samples.

### Upload

```
PUT /v2/sessions/{id}/pcm?offset=<sample>
```

Any range, in any order, in the session's `format`. Send the cue region first.
The track is hashed, shared and stored once every sample has arrived. A sample
sent twice must match to the bit (`409` otherwise). Past the track end is `400`;
a closed session is `410`.

### Window

```json
PUT /v2/sessions/{id}/window
Content-Type: application/json

{"regions": [
  {"at": 661500,  "ahead": 705600, "behind": 88200, "priority": 0},
  {"at": 2205000, "ahead": 352800, "priority": 60}]}
```

A region asks for the segments covering `at − behind .. at + ahead`. `at` and
`priority` are required; `ahead` and `behind` default to `0`. Up to 32 regions;
`{"regions": []}` asks for nothing.

| `priority` | Runs |
| --- | --- |
| `0` to `49` | before whole-track jobs, also between two segments of one |
| `50` to `99` | only when no job is waiting |

- Segments run by priority, then by distance from their region's `at`, behind
  counted twice. Across sessions the same order holds; at a tie the session
  served longest ago goes first.
- A window replaces this session's regions only. The server never moves a
  region: send a new window when the needle or the cues move.
- A running segment finishes. A segment already run or stored does not run
  again.

Answers the view. `400` for a malformed body, a priority over 99 or more than 32
regions, `410` when the session is closed.

To have audio ready where the needle will be, lead it by the time two segments
take:

```text
at = needle + (w + 2 × segment_run_seconds) × play rate × sample_rate
w  = 0 when the worker is idle, else up to one segment_run_seconds
```

### View

```json
GET /v2/sessions/{id}
{"id": "s1-0a1b2c3d", "state": "open", "rev": 7212845571041349902,
 "track": {"id": "f9c29cbfe3466b674f9903d9", "frames": 13230000,
           "sample_rate": 44100, "channels": 2},
 "output": {"format": "flac", "sample_rate": 96000, "gain": 0.7071068,
            "parts": ["harmonics", "vocals"], "derived": "drums"},
 "grid": {"segment_frames": 343980, "stride_frames": 257985, "segments": 52,
          "segment_run_seconds": 0.41},
 "arrived": [[0, 13230000]],
 "ready": [[0, 1031940]],
 "stems": [
   {"name": "harmonics", "sources": 6, "stored": "whole", "bytes": 28114522},
   {"name": "vocals", "sources": 8, "stored": "pieces", "pieces": [[0, 4], [7, 9]]}],
 "complete": false, "planned": 6, "idle_timeout_secs": 30}
```

| Field | |
| --- | --- |
| `state` | `open`, or `closed` with `error` saying why. |
| `rev` | Changes when any other field does. A u64: compare the `ETag` instead in JavaScript. |
| `track.id` | The track id, `null` until every sample has arrived. |
| `grid` | Segment length and step, in samples; see [Pieces](#pieces). |
| `arrived`, `ready` | Half-open `[from, to]` spans uploaded, and separated in memory. |
| `stems[].stored` | `whole` with `bytes`, `pieces` with half-open piece index spans in `pieces`, or `none`. |
| `stems[].sources` | Bitmask of the sources a stem holds: `1` drums, `2` bass, `4` other, `8` vocals. |
| `complete` | Every sample is ready or stored: any range reads, the whole stem included. |
| `planned` | Segments asked for and not started. |
| `waiting_for` | Present while planned segments wait: `upload`, or `the whole upload, or mean and std`. |

The view carries `ETag: "<rev in hex>"`. Long-poll by sending it back:

```
GET /v2/sessions/{id}?wait=25
If-None-Match: "64192cd4e1ab510e"
```

The server answers when the view changes, or `304` after `wait` seconds (at most
25, and at most half of `idle_timeout_secs`). Every request keeps the session
open.

### Read

```
GET /v2/sessions/{id}/stems/{name}?from=<sample>&to=<sample>    a range
GET /v2/sessions/{id}/stems/{name}                              the whole stem
```

`from` and `to` are track samples; send both or neither. The body is in the
session's `output_format`, at the output rate.

| Header | |
| --- | --- |
| `x-stemd-from` | First sample of the body, at the output rate. |
| `x-stemd-frames` | Samples in the body. |
| `x-stemd-gain` | The session gain. |
| `x-stemd-clipped` | Samples clipped to fit 16 bits. Absent on a stored piece sent as it is. |

| Status | |
| --- | --- |
| `200` | The range. |
| `400` | Only one of `from` and `to`. |
| `404` | No such session. |
| `409` | A sample is neither ready nor stored, the range is empty or past the end, the stem is not served, or the session is closed. |

- A range is read from memory where it is ready there, and from stored files
  otherwise. At the model's rate both give the same samples. At another rate,
  about 1 sample in 10,000 can differ by 1 LSB, since stored pieces were
  converted one by one.
- The first whole-stem read stores the stem whole; later ones send that file.
- Reads run beside the worker and do not slow separation.

### Pieces

Piece `k` is track samples `k × stride_frames` to `(k + 1) × stride_frames`, the
last one cut at the track end. Piece `k` is ready once segments `k − 1` and `k`
have run.

```
GET /v2/sessions/{id}/pieces/{name}/{k}
```

The piece, encoded, with the [read headers](#read). `404` for `k` past the last
piece or a stem the session does not serve; `409` when it is not ready or
stored. When the stem is stored whole, the answer also carries
`x-stemd-whole: /v2/sessions/{id}/stems/{name}`: fetch that once instead of
the remaining pieces.

| Segments | Fast, Balanced | Quality |
| --- | --- | --- |
| length | 343,980 samples (7.8 s) | 588,800 samples (13.35 s) |
| step | 257,985 samples | 441,600 samples |

Read the values from `grid`.

### Storage

- Pieces are stored per track id, model and output, under `--pieces-dir`.
- Once every piece of a track is stored, the server rewrites each stem as one
  whole file in the background and deletes its pieces. A whole-track job or a
  whole-stem read does the same right away.
- Past `--pieces-max-gb` the least recently used track is evicted. A track an
  open session plays is never evicted.
- A piece missing from disk shows in `stems` and turns `complete` to `false`;
  send the window again to run it.

### End

`DELETE /v2/sessions/{id}` (`204`), or let it go idle: a session with nothing
to run and no request for `idle_timeout_secs` is ended. A model switch ends
every session. An ended session's id is `404`.

## Separations

What the server holds for one track and output, without uploading:

```
GET /v2/separations/<track id>?output_format=flac&output_sample_rate=96000&parts=harmonics,vocals
```

The parameters are the output parameters of a job. `mp3` is `400`; a track id
that is not 24 lowercase hex is `400`. `GET /v2/separations?file=<key>&<output>`
finds the track by [file key](#file-key) instead, and answers `404
unknown_separation` for a key the server has not recorded or found ambiguous.

```json
{"track": "f9c29cbfe3466b674f9903d9", "model_id": "htdemucs-22c7f2c9",
 "frames": 13230000, "sample_rate": 44100, "channels": 2,
 "output": {"format": "flac", "sample_rate": 96000, "gain": 0.7071068,
            "parts": ["harmonics", "vocals"], "derived": "drums"},
 "state": "stored",
 "stems": [
   {"name": "harmonics", "sources": 6, "stored": "whole", "bytes": 28114522},
   {"name": "vocals", "sources": 8, "stored": "whole", "bytes": 9120333}]}
```

| `state` | The server holds | The client |
| --- | --- | --- |
| `stored` | every stem whole | fetches the stems or the bundle |
| `derivable` | every stem whole in another 16-bit container, or in `pcm32` at the default gain; `from` is that output block | fetches: the server converts once, stores, sends |
| `consolidating` | every piece; whole files are being written | asks again shortly, or joins with a session |
| `open` | the mix, in a session | opens a session with `track=<id>` |
| `partial` | some pieces or whole files | opens a session and uploads |

`404 unknown_separation` means nothing is held for this output.

- `frames` and `sample_rate` are at the model's rate.
- `output.gain` is the stored gain, any gain matches; it is absent for `open`.
- For `derivable`, every `stems[].stored` is `none`.
- A track whose whole files fail to write reads as `partial` and is retried
  after 10 minutes.

### Stems and bundle

```
GET /v2/separations/<track id>/stems/<name>?<output parameters>
GET /v2/separations/<track id>/bundle?<output parameters>
```

`200` when the state is `stored` or `derivable`, `409` otherwise, `404
unknown_separation` when nothing is held, `410` when the stems are evicted
during the request: ask again. A stem carries the [read headers](#read). A bundle carries `x-stemd-frames` and is
`application/octet-stream`.

The bundle is an index, then each stem's file as `/stems/<name>` sends it, back
to back. Little-endian:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `STMB` |
| 4 | 2 | version `u16`, `1` |
| 6 | 2 | format `u16`: `0` flac, `1` wav, `2` pcm16, `3` pcm32 |
| 8 | 4 | stem count `n`, `u32` |
| 12 | 4 | frames at the output rate, `u32` |
| 16 | 4 | output rate, `u32` |
| 20 | 4 | channels, `u32` |
| 24 | 32 × n | one entry per stem, in [stem order](#parts) |

| Entry offset | Size | Field |
| --- | --- | --- |
| 0 | 2 | sources `u16`, the bitmask above |
| 2 | 2 | reserved, `0` |
| 4 | 4 | gain `f32` |
| 8 | 8 | offset of the stem's file from the start of the bundle, `u64` |
| 16 | 8 | length of the stem's file, `u64` |
| 24 | 4 | samples clipped `u32`; `0xFFFFFFFF` for pcm32 |
| 28 | 4 | reserved, `0` |

A player skips an entry whose `sources` holds a bit it does not know.

## A deck loading a track

1. Compute the [file key](#file-key). With `model_id`, the key and the output in
   the client's own cache, stop here.
2. `GET /v2/separations?file=<key>&<output>`. On `stored` or `derivable`, fetch
   the bundle and stop. On `open`, open a session with `track=<track>`.
3. Otherwise open a session with `file=<key>` and the method `/v2/health`
   allows, send a window, long-poll the view, read ranges near the needle.
4. When the view says `complete`, fetch each stem whole, or the bundle once the
   lookup says `stored`. Store the bytes as they are, keyed on `model_id`, the
   track id, the format and the rate. End the session.

Store the shipped parts losslessly and rebuild the derived one on playback.

## Discovery

The server advertises `_stemd._tcp` over mDNS:

```
features=  model=htdemucs  path=/v2  v=2  version=0.2.1
```

| Entry | |
| --- | --- |
| `v` | Protocol version. Accept only versions you speak. |
| `path` | Where the API lives. |
| `features` | As in `/v2/health`. |
| `model` | The artefact name, a label only: cache on `model_id`. |
| `version` | Server version. |

| Protocol | Path | Servers |
| --- | --- | --- |
| 1 | `/v1` | 0.2.0 and earlier; deprecated, served beside protocol 2 |
| 2 | `/v2` | 0.2.1 and later |

Any change to a route, a field or a default in this document is protocol 3.

## Protocol 1 (deprecated)

`/v1` keeps the 0.2.0 health and jobs for clients that have not moved to
protocol 2, such as stemd-cli 0.2.0. It runs on the same jobs and cache as
`/v2`. It is deprecated and may stop working in a future release. Every `/v1`
answer carries:

```
Deprecation: true
Link: <https://github.com/nsaintot/stemd/blob/main/docs/api.md>; rel="deprecation"; type="text/html"
```

The server logs one warning per client address that calls `/v1`, and the
window's Clients list marks those jobs `protocol 1`.

| Route | Protocol 1 |
| --- | --- |
| `GET /v1/health` | `stems`, `default_output_format`, `default_output_sample_rate`, `derived_stem` and `default_parts` in place of `default_output`. No `features`, `session` or `segment_run_seconds`. |
| `POST /v1/jobs`, `GET /v1/jobs/{id}` | The result has `sample_rate`, `format` and `derived_stem` in place of `output`, and each stem has its server-local `path`. No `file=`. |
| `GET /v1/jobs/{id}/stems/{name}`, `DELETE /v1/jobs/{id}`, `GET /v1/logs` | As under `/v2`. |
| `/v1/sessions` and every path below it | `410` with code `protocol_1`. Sessions are protocol 2 only: use `/v2/sessions`. |

Nothing else is served under `/v1`. The TXT record advertises protocol 2 only.

## Reference client

```bash
cargo run --release -p stemd-cli -- track.mp3
cargo run --release -p stemd-cli -- track.wav --host 10.10.50.245:8420 --out ./stems
```

`stemd-cli` finds a server over mDNS or takes `--host`, decodes the file,
submits a job, rebuilds the derived part, and checks that the three parts sum to
the mix. `--f32` uploads and receives `f32`, `--dsp-filter <file>` uploads and
uses a filter, `--include-derived` asks the server for the third part.
