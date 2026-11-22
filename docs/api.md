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
POST   /v2/jobs/{id}/join                          claim a running job found by lookup
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

`code` is present only on errors that a client acts on:

| `code` | Status | The client |
| --- | --- | --- |
| `unknown_filter` | `404` | uploads the filter, then resends |
| `unknown_separation` | `404` | opens a session and uploads, or submits a job |
| `protocol_1` | `410` | uses `/v2/sessions`: sessions are on `/v2` only |

A malformed query or JSON body gets a plain-text `400`, `415` or `422`; a
malformed window gets a JSON `400`. Every route, `/v2/health` included, answers
`503` while the server is switched off in its window.

### Health

`GET /v2/health`:

```json
{"version": "0.2.1", "backend": "mlx", "model": "htdemucs", "device": "gpu",
 "sample_rate": 44100, "channels": 2, "preset": "Fast",
 "model_id": "htdemucs-22c7f2c9",
 "completed_jobs": 0, "queue_depth": 0,
 "cached_tracks": 0, "cached_bytes": 0, "cache_max_bytes": 8000000000,
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
| `model_id` | The model that produced the stems. Key a cache on it; see [Identifiers](#identifiers). |
| `completed_jobs` | The number of jobs the server still answers for, finished or not. |
| `running_job` | The id of the job running now; absent when the server is idle. |
| `cached_tracks`, `cached_bytes`, `cache_max_bytes` | The tracks and bytes the [cache](#cache) holds on disk, and its byte cap. |
| `max_track_seconds` | The longest track the server accepts. A longer track gets `413`. |
| `output_sample_rates` | The values that `output_sample_rate` accepts. |
| `dsp_modes`, `dsp_mode_pairs` | The values that `dsp_mode` accepts, and the one rate pair that each numbered mode covers. |
| `parts` | The parts a request can list in `parts`. |
| `default_output` | The output a request gets when it names no format, rate or parts. The server's owner sets it in the window, and clients read it here. |
| `features` | Behaviors added after protocol 2. The list is empty for now. |
| `segment_run_seconds` | The median time in seconds that the loaded model took per segment, over the last 16 segments. It is `null` until a segment has run, and again after a model switch. |
| `session` | The [upload methods](#upload-methods) a session accepts; `null` when the model cannot run segment by segment. |

Read `/v2/health` again after a model switch: `model_id`, `preset`,
`segment_run_seconds` and `session` change with the model.

### The output block

Job results, session views and lookups describe their stems with the same block:

```json
"output": {"format": "flac", "sample_rate": 44100, "gain": 0.7071068,
           "parts": ["harmonics", "vocals"], "derived": "drums"}
```

| Field | |
| --- | --- |
| `format` | `flac`, `wav`, `mp3`, `pcm16` or `pcm32`. |
| `sample_rate` | The output sample rate. |
| `gain` | Multiply every sample by `1.0 / gain` to restore the original level. It is absent in job results, where each stem carries its own gain, and for an open track. |
| `parts` | The parts the server ships. |
| `derived` | The part the client rebuilds as `mix` minus the shipped parts, or `null` when no part is left out. |

### Cache

The server keeps every separation from jobs, sessions and window drops in one
cache that survives restarts, keyed by model, output and track id. A job finds
the stems a session stored, and a session finds the stems a job stored.

| A track is | |
| --- | --- |
| in use | played by an open session, or part of a job result with a stem that was not fetched within `--unfetched-ttl` (300 s) of the job ending or of the last fetch |
| whole | every stem is stored as one file |
| partial | at least one stem is still stored in pieces |

| Rule | Default |
| --- | --- |
| A track in use is never deleted. | |
| A partial track unused for `--partial-max-hours` is deleted. | 24 |
| A whole track unused for `--whole-max-days` is deleted; `0` keeps whole tracks forever. | 30 |
| When the cache exceeds `--cache-max-gb`, or the free disk space minus 2 GB, partial tracks are deleted first, oldest first, then whole tracks, least recently used first. | 8 |

The window's Settings sheet sets the cache size and the whole-track age and
saves both for the next launch; a flag overrides either one for a single run.
`--ephemeral` empties the cache at start and at quit.

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
- **Job `id`**: the first part is the start of the track id, and the attempt
  number keeps two clients' jobs on the same track apart.

### Track id

The track id is a SHA-256 hash over:

1. three fields, each a little-endian `u32` byte length then its UTF-8 bytes:
   `stemd-track-1`, the sample rate (`44100`), the channel count (`2`);
2. every sample as little-endian `f32`, interleaved, at the model's rate.

The id is the first 24 hex characters of the digest. An `s16le` sample `v`
hashes as the `f32` `v / 32768.0`, so `s16le` and `f32le` uploads of the same
audio share an id.

### File key

Computing the track id needs the whole file decoded. A file key finds a track
without decoding: the client derives it from the encoded file, for example from
its size, its decoded length and a sample of its bytes. The server never
computes it; it only stores and matches it.

- Send `file=<key>` on `POST /v2/sessions` or `POST /v2/jobs`. Once the server
  has hashed the whole upload, it records `key → track id` and persists the
  mapping with the stored stems.
- `GET /v2/separations?file=<key>&<output>` then answers like a lookup by that
  track id, and includes `track`.
- When a key arrives with a second, different track id, it becomes ambiguous:
  lookups by that key answer `404 unknown_separation`, and clients upload the
  track.
- Clients that share a server share entries only when they compute the key the
  same way and their decoders give the same samples. A decoder that trims
  delay or padding differently gives a different track id.

### Track statistics

Sessions take three optional values computed over the whole decoded track, at
the model's rate. The server checks `mean` and `std` against its own
values, bit for bit; on a mismatch it never shares or stores the track.

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

Uploads are raw interleaved PCM:

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
- `pcm32` has headroom: its gain is always `1.0`. In a 16-bit format, a sample
  that is still past full scale after the gain is clipped and counted in
  `x-stemd-clipped`.
- `mp3` suits stems that someone imports or archives. A deck should not use it:
  moving a fader unmasks the coding noise, and lossy parts do not sum to the
  mix. `mp3` goes up to 48 kHz (`96000` with `mp3` gets `400`), and sessions
  refuse it.
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
- When a request asks for `bass` or `other`, `Balanced` runs a third model,
  which takes about 1.5x the time.

### Rebuilding the derived part

Do not store it. Mix it on the fly from the decoded mix and the stems:

```text
out = d·mix + (h − d)·H + (v − d)·V        h, v, d: the three faders
```

At `h = v = d = 1.0` the output is the mix, bit for bit. Snap the faders to a
literal `1.0` to get that. The sum is exact in every lossless format, since the
derived part comes from the same mix it is summed against.

## Jobs

### Submit

```
POST /v2/jobs?sample_rate=44100&channels=2&format=s16le     body: the whole track
```

| Parameter | Default | |
| --- | --- | --- |
| `sample_rate`, `channels`, `format` | `44100`, `2`, `s16le` | The format of the upload. |
| `output_format` | `default_output.format` | See [Output formats](#output-formats). |
| `output_sample_rate` | `default_output.sample_rate` | One of `output_sample_rates`. |
| `parts` | `default_output.parts` | Comma-separated, see [Parts](#parts). |
| `include_derived` | `false` | `true` ships the derived part too. |
| `dsp_mode` | `0` | The output rate converter. `0` handles any pair; numbered modes are deprecated. |
| `dsp_filter` | none | The id of an [uploaded filter](#filters) to convert with, in place of `dsp_mode`. |
| `file` | none | The client's [file key](#file-key), recorded once the upload is hashed. |

| Status | |
| --- | --- |
| `202` | The job is accepted; poll it. A request that matches a running job returns that job. |
| `200` | The stems were cached: the job is already done and no model ran. |
| `400` | A parameter is invalid; `error` names it. |
| `404` | `unknown_filter`. |
| `413` | The track is longer than `max_track_seconds`, or the body is over the upload limit. |
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
              "bytes": 2216440, "gain": 0.7255},
             {"name": "vocals", "url": "/v2/jobs/a8e29ff15c16-1/stems/vocals",
              "bytes": 1120333, "gain": 0.7255}],
   "model_residual_db": -35.2, "separation_secs": 1.39, "realtime_factor": 21.6,
   "cached": false}}
```

- `stage` goes through `queued`, `analysing`, `separating`, `reconstructing` and
  `writing`, and ends at `done`, `failed` (with `error`) or `cancelled`.
- `fraction` runs from 0 to 1, and separation covers 0.10 to 0.90. While the job
  is `queued`, `completed` and `total` give its queue position, and `detail`
  states it in words (`next`, `2 ahead`).
- `result` is present once the job is `done`. `frames` counts samples at the
  output rate. Every stem has the same `gain`: 3 dB over the mix's peak, as for
  a session opened with `peak` (see [Gain](#gain)). A job answered from the
  cache has the gain the stems were stored at.
- `model_residual_db` is a diagnostic: the level of `mix − sum(model sources)`.
  It is around −30 dB on `Fast`, where the residual ends up in the rebuilt
  `drums`, and about −160 dB on `Balanced` and `Quality`, which build one part
  as the remainder. Do not compare it across presets.
- `separation_secs` is the time taken by the run that made the stems; for a
  cached answer, that run was an earlier one. For stems a session stored,
  `separation_secs` is `0` and `model_residual_db` is `null`.

### Download and release

`GET /v2/jobs/{id}/stems/{name}` sends the stem in `output_format`.

| Status | |
| --- | --- |
| `200` | The stem. |
| `404` | The job or the stem does not exist. |
| `409` | The job has not finished. |
| `410` | The stems were evicted from the cache; submit the job again. |

`DELETE /v2/jobs/{id}` releases the job (`204`). The last client to release a
running job cancels it. A finished job answers for `--unfetched-ttl` (300 s)
after it ended or after its last stem fetch, and answers `404` after that;
submitting the track again finds the stems in the cache.

A finished job also stores its stems for [lookups](#separations) and sessions on
the same track.

### Join

`POST /v2/jobs/{id}/join`, with no body, adds the caller as one more client of a
job, such as the job a [lookup](#separations) reports as `separating`. Another
client's `DELETE` then leaves the job running. Release it with `DELETE` like a
job you submitted.

| Status | |
| --- | --- |
| `200` | The job, as `GET /v2/jobs/{id}` answers it, at any stage. |
| `404` | The job does not exist. |

### Filters

A client that rebuilds the derived part at another rate must resample its mix
through the same filter as the stems. It uploads its converter once and names
it in `dsp_filter`:

```
PUT /v2/filters/<id>     body: the filter file; 201 when new, 200 when stored already
GET /v2/filters/<id>     the summary, or 404 unknown_filter
```

`<id>` is the lowercase hex SHA-256 of the file. Both answers carry `{"id",
"in_rate", "out_rate", "taps", "branch_taps"}`. The file layout, all
little-endian:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `SFIR` |
| 4 | 4 | version `u32`, `1` |
| 8 | 4 | input rate `u32`: the model's `sample_rate` |
| 12 | 4 | output rate `u32`: one of `output_sample_rates`, not the input rate |
| 16 | 4 × n | prototype taps, `f32` |

The filter is a stateless polyphase FIR. With `L / M = out / in` in lowest terms
and `n = L × B` taps `p`, output frame `m` is `y[m] = Σ x[k] · p[m·M − k·L]`,
with `p` and `x` zero outside their range. The output has `ceil(frames · L / M)`
frames. The server applies the taps as given (unity gain means they sum to `L`)
and does not compensate any delay. `B` is at most 1024 and `n` at most
1,048,576.

- A malformed file, or a file whose hash is not `<id>`, gets `400`; an oversized
  file gets `413`.
- The server keeps 64 filters across restarts and evicts the least recently used
  one. A job naming an evicted filter gets `404 unknown_filter`.
- A job with `dsp_filter` must ask for exactly the filter's rate pair, and may
  not also set a non-zero `dsp_mode`.

## Sessions

A session separates one track segment by segment, in the order its window asks.
Positions are samples at the model's rate.

```text
open ──▶ upload ──▶ window ──▶ long-poll the view ──▶ read ranges ──▶ end
```

Two sessions on the same audio share one track, and with it the model work, the
memory and the stored pieces. Finished pieces are stored under the [track
id](#track-id) and serve any later session or lookup.

### Open

```
POST /v2/sessions?frames=13230000&format=f32le&output_format=flac
```

| Parameter | |
| --- | --- |
| `frames` | The track length in samples. Required unless `track` is set. |
| `track` | A track id. The session joins an `open` track, which takes no upload. For a `partial` or `consolidating` track, the session opens on the stored pieces, which are ready at once, and the client uploads the rest with `PUT /pcm`. Upload the whole track to have the new pieces stored. |
| `file` | Optional. The client's [file key](#file-key). |
| `sample_rate`, `channels`, `format` | The format of the upload. The rate and channel count must match the model's. |
| output parameters | The same as for jobs, except that `mp3` is refused. |
| `mean`, `std` | Optional; send both or neither. See [Upload methods](#upload-methods) and [Track statistics](#track-statistics). |
| `peak` | Optional. It sets the session's gain; see [Gain](#gain). |

The body is optional audio starting at sample 0; it must be empty when joining
an `open` track.

A `track` id on stored pieces is a claim:

- The server trusts it for reading the stored pieces, and the session takes
  their gain.
- New pieces are stored, and the track is shared, only once the whole upload
  hashes to the id.
- An upload that hashes to another id ends the session with
  `upload is not track <id>`, stores nothing, and makes the file key ambiguous.

Opening a session reserves `(1 + parts) × channels × frames × 4` bytes for the
mix and the parts. Together, open tracks may reserve up to half of physical
memory; joining a track reserves nothing.

| Status | |
| --- | --- |
| `201` | The [session view](#view). |
| `400` | A parameter is invalid: the wrong rate or channel count, `frames=0`, only one of `mean` and `std`, or `frames` that differs from the track that `track` names. |
| `404` | `unknown_separation`: nothing is open or stored for `track`. Open a session with `frames` and upload. |
| `409` | The loaded model cannot run segment by segment. |
| `413` | The track is longer than `max_track_seconds`, or it needs more memory than open tracks may reserve. |
| `429` | The session limit is reached, or the open tracks reserve too much memory for this one. |

### Upload methods

`session.methods` in `/v2/health` says what each way of uploading needs on the
loaded model:

| Method | The open body | Segments run before the upload ends when |
| --- | --- | --- |
| `whole` | the whole track | always; `requires` is empty |
| `partial` | a prefix, or nothing; the rest of the track follows with `PUT /pcm` | the open sent every name in `requires` |

| Model | `partial` requires |
| --- | --- |
| `htdemucs`, `htdemucs_ft` (Fast, Balanced) | `mean`, `std` |
| `stemd_roformer` (Quality) | nothing |

Without the required values a `partial` session still works, but no segment runs
until the last sample arrives. Each name in `requires` and `optional` is an open
parameter; a value whose definition changes gets a new name. A client:

1. ignores keys it does not know;
2. uses a method only when it knows every name in its `requires`, and otherwise
   uses `whole`;
3. ignores methods it does not know;
4. leaves out an optional parameter it cannot compute exactly.

### Gain

A session has one gain, fixed when it opens: 3 dB over `peak`, or over full
scale when the open sends no `peak`.

```text
gain = 1 / (peak × √2)   when that is below 1, else 1      (pcm32: always 1)
```

Without `peak` a 16-bit session ships at `0.7071068`. A stem that still passes
full scale is clipped, and `x-stemd-clipped` counts the samples.

### Upload

```
PUT /v2/sessions/{id}/pcm?offset=<sample>
```

The body is any range of the track, sent in any order, in the session's
`format`. Send the cue region first. The track is hashed, shared and stored once
every sample has arrived. A sample sent twice must match bit for bit, or the
server answers `409`. A range past the track end gets `400`, and an upload to a
closed session gets `410`.

### Window

```json
PUT /v2/sessions/{id}/window
Content-Type: application/json

{"regions": [
  {"at": 661500,  "ahead": 705600, "behind": 88200, "priority": 0},
  {"at": 2205000, "ahead": 352800, "priority": 60}]}
```

A region asks for the segments covering `at − behind .. at + ahead`. `at` and
`priority` are required; `ahead` and `behind` default to `0`. A window holds up
to 32 regions, and `{"regions": []}` asks for nothing.

| `priority` | Runs |
| --- | --- |
| `0` to `49` | before whole-track jobs, and also between two segments of a running job |
| `50` to `99` | only when no job is waiting |

- Segments run in order of priority, then of distance from their region's `at`,
  where a distance behind `at` counts twice. The same order holds across
  sessions; on a tie, the session served longest ago goes first.
- A window replaces this session's regions only. The server never moves a
  region: send a new window when the needle or the cues move.
- A new window lets the running segment finish. A segment that has already run
  or is stored does not run again.

The answer is the [view](#view). A malformed body, a priority over 99 or more
than 32 regions gets `400`; a closed session gets `410`.

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
| `rev` | A u64 that changes whenever any other field changes. In JavaScript, compare the `ETag` instead. |
| `track.id` | The track id, `null` until every sample has arrived. |
| `grid` | Segment length and step, in samples; see [Pieces](#pieces). |
| `arrived`, `ready` | Half-open `[from, to]` spans: `arrived` lists the uploaded samples, `ready` the samples separated in memory. |
| `stems[].stored` | `whole` with the file size in `bytes`, `pieces` with half-open spans of piece indexes in `pieces`, or `none`. |
| `stems[].sources` | Bitmask of the sources a stem holds: `1` drums, `2` bass, `4` other, `8` vocals. |
| `complete` | Every sample is ready or stored, so any range can be read, the whole stem included. |
| `planned` | The number of segments the window asks for that have not started. |
| `waiting_for` | Present while planned segments wait, and names what they wait for: `upload`, or `the whole upload, or mean and std`. |

A span `[from, to]` is half-open: `from` is included and `to` is excluded.

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

`from` and `to` are positions in track samples; send both or neither. The body
is in the session's `output_format`, at the output rate.

| Header | |
| --- | --- |
| `x-stemd-from` | The position of the body's first sample, at the output rate. |
| `x-stemd-frames` | The number of samples in the body. |
| `x-stemd-gain` | The session gain. |
| `x-stemd-clipped` | The number of samples clipped to fit 16 bits. Absent when a stored piece is sent as it is. |

| Status | |
| --- | --- |
| `200` | The range. |
| `400` | Only one of `from` and `to` was sent. |
| `404` | The session does not exist. |
| `409` | A sample is neither ready nor stored, the range is empty or past the end, the stem is not served, or the session is closed. |

- The server reads a range from memory where it is ready there, and from stored
  files elsewhere. At the model's rate both give the same samples. At another
  rate, about 1 sample in 10,000 can differ by 1 LSB, since each stored piece
  was converted on its own.
- The first read of a whole stem stores it as one file; later reads send that
  file.
- Reads run alongside the worker and do not slow down separation.

### Pieces

Piece `k` covers track samples `k × stride_frames` to `(k + 1) × stride_frames`;
the last piece stops at the track end. Piece `k` is ready once segments `k − 1`
and `k` have run.

```
GET /v2/sessions/{id}/pieces/{name}/{k}
```

The answer is the encoded piece with the [read headers](#read). A `k` past the
last piece, or a stem the session does not serve, gets `404`; a piece that is
neither ready nor stored gets `409`. When the stem is stored whole, the answer
also carries `x-stemd-whole: /v2/sessions/{id}/stems/{name}`; fetch that URL
once in place of the remaining pieces.

| Segments | Fast, Balanced | Quality |
| --- | --- | --- |
| length | 343,980 samples (7.8 s) | 588,800 samples (13.35 s) |
| step | 257,985 samples | 441,600 samples |

Read the actual values from `grid`.

### Storage

- Pieces are stored in the [cache](#cache), per track id, model and output.
- Once every piece of a track is stored, the server rewrites each stem as one
  whole file in the background and deletes its pieces. A whole-track job or a
  whole-stem read does the same right away.
- A piece missing from disk shows up in `stems` and sets `complete` to `false`;
  send the window again to run it again.

### End

End a session with `DELETE /v2/sessions/{id}` (`204`), or let it go idle: the
server ends a session that has nothing to run and gets no request for
`idle_timeout_secs`. A model switch ends every session. Requests to an ended
session get `404`.

## Separations

A lookup tells a client what the server holds for one track and output, before
the client uploads anything:

```
GET /v2/separations/<track id>?output_format=flac&output_sample_rate=96000&parts=harmonics,vocals
```

The parameters are the output parameters of a job. `mp3` gets `400`, and so does
a track id that is not 24 lowercase hex characters. `GET
/v2/separations?file=<key>&<output>` finds the track by [file key](#file-key)
instead, and answers `404 unknown_separation` for a key that the server has not
recorded or has found ambiguous.

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
| `derivable` | every stem whole in another 16-bit container, or in `pcm32` at the default gain; `from` is that output block | fetches the stems; the server converts them once, stores the result and sends it |
| `consolidating` | every piece; whole files are being written | asks again shortly, or joins with a session |
| `separating` | a job for this output is running; `job` is its id | posts `/v2/jobs/<job>/join`, follows the job, fetches its stems when it is done and deletes it when leaving; on 404, 410, `failed` or `cancelled`, looks up again |
| `open` | the mix, held by an open session | opens a session with `track=<id>` |
| `partial` | some pieces or whole files | opens a session with `track=<id>` and uploads |

`404 unknown_separation` means the server holds nothing for this output.

- `frames` and `sample_rate` are at the model's rate.
- `output.gain` is the gain the stems were stored at; a lookup matches stems at
  any gain. It is absent for `open` and `separating`.
- For `derivable`, every `stems[].stored` is `none`.
- When writing a track's whole files fails, the track reads as `partial`, and
  the server retries after 10 minutes.

### Stems and bundle

```
GET /v2/separations/<track id>/stems/<name>?<output parameters>
GET /v2/separations/<track id>/bundle?<output parameters>
```

Both routes answer `200` when the state is `stored` or `derivable`, `409` in any
other state, `404 unknown_separation` when nothing is held, and `410` when the
stems are evicted during the request, in which case the client asks again. A
stem carries the [read headers](#read). A bundle carries `x-stemd-frames` and
has the content type `application/octet-stream`.

A bundle is an index followed by each stem's file, back to back, with the same
bytes that `/stems/<name>` sends. All fields are little-endian:

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

1. Compute the [file key](#file-key). If the client's own cache holds this
   `model_id`, key and output, stop here.
2. `GET /v2/separations?file=<key>&<output>`. On `stored` or `derivable`, fetch
   the bundle and stop. On `separating`, join `/v2/jobs/<job>` and follow it. On
   `open`, open a session with `track=<track>`. On `partial`, open one with
   `track=<track>` and `file=<key>`, and upload the whole track with `PUT /pcm`.
3. Otherwise, open a session with `file=<key>` and an upload method that
   `/v2/health` allows, send a window, long-poll the view, and read ranges near
   the needle.
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
| `v` | The protocol version. Accept only versions the client speaks. |
| `path` | The path prefix of the API. |
| `features` | The same list as `features` in `/v2/health`. |
| `model` | The artefact name. It is only a label; key a cache on `model_id`. |
| `version` | The server version. |

| Protocol | Path | Servers |
| --- | --- | --- |
| 1 | `/v1` | 0.2.0 and earlier; deprecated, served beside protocol 2 |
| 2 | `/v2` | 0.2.1 and later |

Any change to a route, a field or a default in this document needs protocol 3.

## Protocol 1 (deprecated)

`/v1` keeps the 0.2.0 health and jobs routes for clients that have not moved to
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
| `POST /v1/jobs`, `GET /v1/jobs/{id}` | The result has `sample_rate`, `format` and `derived_stem` in place of `output`, and each stem has its server-local `path`. `model_residual_db` is always a number: for stems a session stored, it is `-200` and `realtime_factor` is `0`. No `file=`. |
| `GET /v1/jobs/{id}/stems/{name}`, `DELETE /v1/jobs/{id}`, `GET /v1/logs` | The same as under `/v2`. |
| `/v1/sessions` and every path below it | `410` with code `protocol_1`. Sessions are protocol 2 only: use `/v2/sessions`. |

Nothing else is served under `/v1`. The TXT record advertises protocol 2 only.

## Reference client

```bash
cargo run --release -p stemd-cli -- track.mp3
cargo run --release -p stemd-cli -- track.wav --host 10.10.50.245:8420 --out ./stems
```

`stemd-cli` finds a server over mDNS or takes `--host`, decodes the file,
submits a job, rebuilds the derived part, and checks that the three parts sum to
the mix. `--f32` uploads and receives `f32`, `--dsp-filter <file>` uploads a
filter and converts with it, and `--include-derived` asks the server for the
third part.
