# Internals

Internals of the server. See [api.md](api.md) for the user-facing API.

## Cache

Jobs, sessions and drops share one cache in `~/Library/Caches/stemd`
(`--cache-dir`), kept across restarts. The code is in `cache/`.

```text
<cache id>/<type>/<track id>/<stem>/<k>.<ext>    a piece
<cache id>/<type>/<track id>/<stem>.<ext>        a whole stem, described by <stem>.json
<cache id>/<type>/<track id>/track.json          the track length and piece grid
<cache id>/<type>/<track id>/.used               the last use, stored as its modification time
<cache id>/<type>/<track id>/.separation.json    a job's residual and separation time
.files/<file key>                                the track id that a file key maps to
```

| Level | |
| --- | --- |
| cache id | `model_id` plus 8 hex characters of the full identity, such as `htdemucs-22c7f2c9-1f2e3d4c` |
| type | format, rate, filter, parts and gain, such as `flac-44100-harmonics.vocals-g0.70710677` |
| track id | the hash of the decoded mix; see [api.md](api.md#track-id) |

- Every setting that changes the bytes is part of the path, so a cache hit
  returns the same bytes a new separation would produce.
- The full identity is `m=<digest>;r=<recipe>;p=f16;o=0.25`: the weights, how
  the preset uses them, `--full-precision` and `--overlap`. New fields are
  appended at the end. Every field after `m=` takes its value from a closed set
  that contains no `;`, so splitting from the right is unambiguous.
  `m=` holds the `--demucs-model` value as given.
- A job or a drop decodes the upload, hashes it into a track id, and looks for
  every stem stored whole for its output, at any gain. A hit runs no model. A
  miss runs the model and stores whole stems at the track's gain, `1/(peak·√2)`,
  which is the gain a session that sends `peak` gets.
- A drop writes its folder at unity gain. A 16-bit stem stored below unity is
  decoded and encoded again, and a stem that would clip is scaled down, with the
  scaling noted in `stems.json`.

### Deletion

| Track state | |
| --- | --- |
| in use | played by an open session, being written by a job, or part of a job result with a stem that was not fetched within `--unfetched-ttl` (300 s) of the job ending or of the last fetch |
| whole | every stem is stored whole, and no piece directory is left |
| partial | any other track |

| Rule | When |
| --- | --- |
| A track in use is never deleted. | always |
| A partial track unused for `--partial-max-hours` (24; `0` keeps it forever) is deleted. | at start and hourly |
| A whole track unused for `--whole-max-days` (30; `0` keeps it forever) is deleted. | at start and hourly |
| Over `--cache-max-gb` (8), or over the free disk space minus 2 GB, partial tracks are deleted first, oldest first, then whole tracks, least recently used first, until the cache is down to 90%. | after each write and every 5 s |

- The cache size and the whole-track age are settings. The window's Settings
  sheet changes them, applies them at once, and saves them in `settings.json` as
  `cache_max_gb` and `whole_max_days`. A flag overrides a setting for the run
  and greys out its control.
- At start, the server also deletes `.part` files and the per-job directories of
  stemd 0.2.0, which are named by 64 hex characters (`migrate.rs`, to be removed
  with protocol 1). `--ephemeral` empties the cache at start and at quit.
- Files are written to a `.part` file and then renamed. A whole stem counts as
  stored once its description file is in place.
- Deleting a file while a client streams it is safe: the handler opens the file
  first, and an unlinked file stays readable. A fetch that has not opened the
  file yet gets `410`.
- A job handle is dropped `--unfetched-ttl` after the job ended or after its
  last stem fetch, or once its stems are gone. A failed job's handle is dropped
  after the same time.

### Joining

Two clients often ask for the same track within seconds, while the first
separation is still running. A request that matches a job in flight joins it,
and both clients get the result together. The job key is the track id, the
output and the cache id. The in-flight check runs before the cache lookup, and
no request can fall between the two checks, because the worker stores the stems
before it marks a job done. A shared job counts its waiters, so one client
cancelling does not cancel the separation for the other.

## Queue

Separation runs one job at a time. One track saturates the GPU, so a second
concurrent job would finish no sooner, would slow both jobs down, and would make
the progress figures meaningless.

The queue gives clients:

- **FIFO order**: jobs complete in the order they were submitted.
- **A live position**, computed when it is read, because positions shift each
  time the worker takes a job.
- **Backpressure**: `429` once `--queue-depth` (16) jobs wait, which bounds
  memory.
- **Cancellation** of the running job and of the waiting jobs.

`DELETE` drops the job handle. Stems already on disk stay in the cache.

When a player skips a track, the running job must be cancelled. Without that, an
abandoned track would hold the next track behind a full separation whose stems
nobody fetches. The backend polls the progress sink between segments, so a
cancel takes effect within one segment, well under a second at the default
preset.

A cancelled job drops its partial overlap-add. Finishing it would cache stems for
a later cue. The next track is more urgent, and a skipped track cued again is
separated again.

The worker applies a model switch between jobs, because the old separator holds
GPU state that a running forward pass uses. The worker also builds the new
separator: MLX weights must be allocated on the thread that evaluates them, so
the switch sends the worker a closure that builds the model, never a loaded
model. The switch thread downloads the weights (up to 672 MB) and then waits for the
worker's answer, so a failed switch can be reported and `/v2/health` never names
a model the worker has not loaded yet.

## API design notes

The reasons behind the contract in [api.md](api.md).

- **Raw PCM uploads.** The client already holds decoded samples. A second
  decoder on the server can disagree on the encoder delay, which leaves a
  shallow null when the client rebuilds the derived part.
- **`model_id`** is `<artefact>-<8 hex>`, where the hex comes from a digest over
  the weights and the preset's recipe. A preset that changes how it uses an
  artefact changes the id, so stems made before the change stop matching.
  Precision and overlap are left out of the id and go only into the server's
  cache id.
- **Numbered `dsp_mode`s and filters.** A client that rebuilds `drums = mix −
  harmonics − vocals` at another rate must resample its mix through the same
  filter as the stems. Two good resamplers agree to about −77 dB of the mix. A
  part 41 dB below the mix then has an error only 36 dB below it. With the same
  filter, the error is −101 dB. A filter's delay is part of the filter: at 96 kHz, mode 1 is 64
  samples later than mode 0.
- **`model_residual_db`.** On `Fast` the four sources come straight from the
  model and miss the mix by the residual, which ends up in the rebuilt part.
  `Balanced` and `Quality` build one part as the remainder, so the residual
  reads about −160 dB and only confirms that the arithmetic is right.
- **Fixed FLAC block size.** STREAMINFO declares `min_blocksize ==
  max_blocksize` and leaves the trailing partial block out of the minimum, as
  libFLAC and ffmpeg do. With unequal sizes, decoders locate frames by
  sample number. Some then stop early or report frames out of order.
- **Read costs** for a 6-minute stem at 96 kHz in FLAC on an M1 Pro:

  | Read | Time | Memory |
  | --- | --- | --- |
  | whole stem, stored whole | the file transfer | a few MB of buffer |
  | whole stem, first read from 62 stored pieces | 1.7 s: decode 0.65 s, encode 1.0 s | 1.5 GB peak, 1.2 GB of it in the FLAC encoder |
  | one piece's range, stored whole | 10 ms plus the encode of the range | the range only |

## Discovery

The server advertises `_stemd._tcp` through the system Bonjour responder
(`DNSServiceRegister` in libSystem, so it adds no dependency). A hardcoded
address does not work on networks without DHCP, where both ends self-assign
link-local addresses that change at each power cycle.

The server withdraws the advertisement on every exit path:

| exit | mechanism |
|---|---|
| window closed / Cmd-Q | `App::on_exit` |
| `SIGINT` / `SIGTERM` | signal task |
| normal return | `Drop` (backstop) |

`Drop` alone is not enough. On macOS, Cmd-Q terminates through `NSApplication`
without unwinding, and a signal does not unwind either, so destructors never
run. The record would then linger until its TTL expired, and clients would try
to reach a dead server for minutes. Withdrawal is idempotent, so the paths can
overlap.

The server logs its network interfaces at startup, at `debug` level, so a
discovery failure on a multi-homed host can be diagnosed from one log line:

```
interfaces: en0=192.0.2.11 en7=198.51.100.4 bridge100=203.0.113.1
```

`--no-mdns` disables the advertisement, and `--instance` sets its name. An
advertisement failure is logged and ignored, since a client configured with the
server's address still works.

## Drops

A file dropped on the window becomes the same job that `POST /v2/jobs` makes,
with the same track id, job table, worker and cache. A drop and a client asking
for the same track join one separation, and dropping the same file twice is a
cache hit. A drop differs from a client job in two ways: the audio comes from a file, and a
drop always asks for the derived part, because a folder of stems has no mix to
rebuild it from.

The GUI owns the main thread on macOS, so the server builds the HTTP runtime
explicitly in both modes instead of using `#[tokio::main]`.

`ui/probe.rs` runs egui headless (`Context::run_ui` with synthetic input) to
test the layout without a window.

## Logging

| Level | Records |
| --- | --- |
| `error` | work that did not happen |
| `warn` | work that happened differently than asked |
| `info` | what the server did to a client's audio; a whole separation logs two `info` lines |
| `debug` | mechanics: timings, bookkeeping, and code paths that did no work |

The two sinks filter separately. The console shows `info` and above, which is
what a headless run should print, and the buffer behind `/v2/logs` also keeps
`debug`. `RUST_LOG` overrides both.

## Settings

The model, output format, output sample rate, port, longest track, cache size
and cache age persist to `~/Library/Application Support/stemd/settings.json`.
Every field is optional and is stored as the string the matching flag would
take, so the file stays editable by hand, and an unreadable field loses only
that field.

A flag applies to one run, so `SettingsFile` keeps two copies: what the file
says and what this run uses. A write serialises the whole document, so with a
single copy, changing the rate would also save the value of an `--output-format`
flag.

The format and the rate form a pair, and `SettingsFile` corrects a pair that
cannot exist before the server uses it. For example, `mp3` has no 96 kHz mode,
and LAME does not refuse that rate: it turns on its own resampler and spends
eighteen seconds per stem producing a 48 kHz file. Flags, the settings UI, a
hand-edited file and an older file can all produce an invalid pair.

The Settings sheet applies changes while the server runs:

| setting | applied by |
| --- | --- |
| port | binding the new port, handing it to the serve loop, and moving the mDNS advertisement; the old listener stops accepting and lets its open requests finish |
| longest track | each request, which reads it and computes its body cap from it |
| cache size and age | `Cache::set_limits`, which deletes the tracks that the new limits exclude |

Memory caps the longest track. A separation peaks at about 1.8 GB plus 155 MB
per minute of audio, and the cap keeps that within three quarters of physical
memory: 29 minutes on 8 GB, 71 on 16 GB and 320 on 64 GB. A saved length or a
flag above the cap is cut to the cap, with a warning.

## Shutdown

Every way of quitting, Cmd-Q included, goes through `shutdown::now`. It stops
the worker (cancelling the running job, which the separator notices between
segments), withdraws the mDNS advertisement, flushes the output streams and
calls `_exit`.

`_exit` skips the C++ static destructors. MLX holds a Metal device and an
allocator in C++ globals, and destructors that run during a separation are a
crash risk. The skipped destructors only free memory that the kernel reclaims
anyway.

## Writing the stems

This stage converts each stem to the output rate, then encodes it. Both steps
run one thread per stem: one per part, plus one for the derived part. Stems
share only the staging directory, and the GPU work is already done. The stage
takes as long as the slowest stem. Times
for a two-minute track with three parts:

| format | rate | serial | concurrent |
|---|---|---|---|
| mp3 | 48 kHz | 7.37 s | **2.56 s** |
| mp3 | 44.1 kHz | 6.80 s | **2.32 s** |
| flac | 96 kHz | 3.10 s | **1.10 s** |
| flac | 44.1 kHz | 1.75 s | **0.63 s** |
| pcm32 | 96 kHz | 1.33 s | **0.61 s** |

WAV and the raw formats need only a conversion and a write, 50 ms for three
stems. They take the same path and gain nothing from it.

The gain from concurrent conversion depends on the rate. Converting to 48 kHz
runs 2.7x faster, although the converter already uses threads. Converting to
96 kHz gains nothing, because it is bound by writing 92 MB per stem. Neither case gets
slower, so conversion runs concurrently at every rate.

LAME runs at its recommended quality, `q=2`. Do not go back to `q=0`: it takes
twenty seconds for a seven-minute stem, so three stems in sequence cost more
than the separation, and it adds no quality. The `q=0` and `q=2` outputs differ
by −47 dB, 5 dB below the coding noise that both add. From `q=0` to `q=7`, the
noise against the source moves by only 0.9 dB, and it is worse at the slow end.
At a constant 320 kbps there are enough bits that the quantisation search has
nothing left to find. Both benchmarks are `#[ignore]`d tests in `stemfmt.rs`;
re-run them after changing the encoder.

`separation_secs` excludes this stage. It reports the model's cost, which a
client uses to judge the separation whatever the encoder settings. The `done`
log line reports the writing time beside it.

## Model loading

`stemd-mlx` builds `htdemucs` layer by layer on MLX. MLX has no equivalent of
TorchScript, so there is no traced graph, and every layer is a place where the
port can differ from the original. Each stage is therefore nulled against the
reference implementation before the next stage is built on it. A whole track
nulls at -123.7 dB on Metal.

A null shows that two implementations of the same arithmetic agree down to the
precision floor they share, and that floor depends on the backend. The -123.7 dB
figure holds for Metal only. `tests/model.rs` asserts the -60 dB bar that every
backend must clear, and a new backend measures its own figures stage by stage,
from `spectral` upward.

The weights have no manifest. The architecture is compiled in as
`htdemucs::Config`, every layer checks the shape of each tensor it reads, and an
artefact with another architecture fails at load with the name of the tensor
that disagreed. The loader counts the models in a file from its tensor names.

**The preset decides which models run.** `Fast` runs one model over the track.
`Balanced` runs two, each supplying the stem it is best at, and builds
`harmonics` as the remainder (`stemd_core::hybrid`;
[evaluation.md](evaluation.md) explains why). `Quality` runs one 4-stem
BS-RoFormer, keeps its drums and vocals, and ships the same remainder
(`stemd_core::roformer`).

A fine-tuned set such as `htdemucs_ft` does not load as a plain model. Each of
its four checkpoints produces a usable version of only one source, so taking all
four sources from one checkpoint would ship four stems that look right and are
wrong. `MlxSeparator` refuses such a set with an error that says so.

Three constraints hold the design together:

- **MLX state belongs to the thread that built it** for its whole life. Metal
  fails only on a second live command encoder, but CUDA keeps its stream
  registry in thread-local storage, so weights allocated on another thread fail
  their first `eval` with `There is no Stream(gpu, N) in current thread`. The
  queue worker constructs its own separator for this reason.
- **Each model runs at the precision that suits it on the backend**: on Metal,
  every model runs at float16; on CUDA, demucs runs at float16 and the
  RoFormer at float32; the CPU runs float32. At float16 the normalisations
  keep their statistics in float32. Without this exception the model produces
  NaN: a group norm with one group sums a million squares. `--full-precision`
  runs every model at float32, which on Metal takes 1.3x the time.
- **Everything around the model stays float32**: the spectrogram and its
  inverse, the per-branch standardisation and the overlap-add across segments.
  These steps cost a few percent of a forward pass, and half precision would
  hurt quality there.

Two choices keep a forward pass fast. Together they take a 120 s track from
15.5 s to 6.5 s.

- The inverse transform does its overlap-add as four reshapes: `n_fft` is four
  hops, so frames four apart do not overlap. Do not place each of the 336 frames
  into a zeroed whole-segment buffer: that costs 40% of a forward pass.
- The normalisations use `fast::layer_norm`, one fused kernel. Written out by
  hand, each one takes eleven passes over the tensor, and a forward pass runs
  sixty-four of them.

The server hashes and decodes the upload on a blocking pool. Hashing a
five-minute upload takes about 160 ms and decoding about 30 ms; on the async
runtime, either would stall the stem downloads of a client that is already
waiting.

## Model artefacts

The weights are 168 to 672 MB per artefact and never change. The server
downloads them on first run from this project's releases
([models-v2](https://github.com/nsaintot/stemd/releases/tag/models-v2),
[models-v3](https://github.com/nsaintot/stemd/releases/tag/models-v3)) into
`~/Library/Application Support/stemd/models`, never into the bundle, whose
contents the code signature covers.

The project mirrors the weights in its own releases. The demucs conversions come
from mlx-community, and the RoFormer conversion has no upstream. If either
publisher retagged or removed a file, every new install would fail, so the
project hosts its own copies.

Downloads are pinned by SHA-256 and verified before the file is moved into
place. The download goes to a temporary file that is then renamed, so an
interrupted run cannot leave a half-written file that looks complete. No option
skips the check, because the server loads the file's tensors and multiplies them
with every sample of the user's audio.

**The cache is re-hashed on every load.** A check at download time covers only
the download: a file can rot afterwards, and the shape checks catch a wrong
artefact but not a flipped bit. Hashing costs about a second, against tens of
seconds per track.

A cached file that fails the check is deleted and downloaded again. Under
`--offline` it is refused and left in place for inspection, since it cannot be
replaced. A mismatch in a file outside the cache is only reported, because a
locally converted artefact is expected to differ from the published one.

The server looks for weights in this order: `--models`, next to the executable
(including `Contents/Resources`), the cache, then the network.

## Bundle

A bundle needs two things that `cargo run` does not:

- **Models are found relative to the executable.** Finder starts a bundle with
  the working directory set to `/`, so a relative `--models` path cannot resolve
  against the working directory.
- **`-psn_*` arguments are discarded.** Finder can pass a process serial number
  when it launches a bundle, and clap would reject it as an unknown flag.

`bundle-app.sh` copies in one executable, `mlx.metallib` and
`Sparkle.framework`. MLX links statically, so the binary depends only on system
frameworks and needs no `install_name_tool` or rpath. Sparkle is loaded at run
time and signed before the bundle. Without a Developer ID the bundle is ad-hoc
signed and launches only on the machine that built it; distribution needs a
Developer ID, the hardened runtime and notarization. `Info.plist` carries
`NSLocalNetworkUsageDescription`, because the app advertises over mDNS and
serves on the LAN, and Sparkle's `SUFeedURL` and `SUPublicEDKey`.

## Updates

`update/` checks for a newer release. The title bar has two update controls:

- The version, centred beside the title. A click checks for updates, and its
  icon spins during the check.
- The update slot, in the corner away from the window buttons (right on macOS,
  left on Windows). It shows "Checking…", "Up to date" or "Check failed" for 2.6
  s after a check, and "New update available!" once an update is found. Its
  gradient moves for 21 s, and after that only under the pointer.

| host | check | slot click | install |
| --- | --- | --- | --- |
| macOS | Sparkle checks `releases/latest/download/appcast.xml` daily; a click on the version runs a probe; the app menu has "Check for Updates…" | opens Sparkle's window | Sparkle replaces the bundle and relaunches |
| Windows | the newest `v*` release, at launch, daily and on a click | asks, then downloads Setup ("Downloading N%") | Setup runs silently once nothing is running, then restarts Stemd |
| Linux | the newest `v*` release, at launch and daily | opens the release page | the package manager |

Linux has no title bar of its own, so "New update available!" shows in the
status bar when it fits. On Windows:

- Setup is checked against the SHA-256 that GitHub records for the asset, and
  checked again before it starts. A download that receives no data for 60 s
  fails.
- The install mode comes from the uninstall key whose `InstallLocation` is the
  executable's folder, looked up per user first, then for all users. With no
  match (the zip build), or with an all-users copy on a standard account, the
  slot opens the release page.
- If Setup cannot start, for example because the user declined consent, the app
  stays open and offers the update again.
- While a separation runs or waits in the queue, or a model downloads, the slot
  says "Update ready". Once the app is idle, it starts Setup with `/S /relaunch`
  and the install mode, then quits; Setup waits until the executable is free.
- Quitting while the slot says "Update ready" starts Setup without `/relaunch`.

macOS reads `releases/latest/download/appcast.xml`, so a model release
(`models-v*`) must be published with `--latest=false`. Windows and Linux pick
the newest `v*` tag themselves.

Setup reuses the choices of the copy it replaces, in silent and interactive
installs:

| choice | remembered by |
| --- | --- |
| install mode | the hive that holds the uninstall key, this account's hive first |
| location | `InstallLocation` in that key |
| GPU support | `GpuSupport` in that key, or for an older copy, `cudart64_13.dll` next to it |

When the CUDA runtime is already usable, the GPU section downloads nothing.

## Tools

`tools/` ships nothing and runs nothing at serve time. It is tracked because it
produced every reference file the null tests read. `tools/eval/make_*.py` dumps
reference tensors stage by stage: the spectrogram, one encoder layer, one
decoder layer, the transformer, a whole segment and a whole track. The tests in
`crates/stemd-mlx/tests/` compare against them in dB.

`pyproject.toml` still lists torch, for weight conversion only. Published model
weights are PyTorch files and stemd runs MLX, so adding a model means opening
the original once to convert its tensors. `tools/export/` does this, and it
produced the RoFormer artefact, since nobody has published an MLX one.
`tools/eval/benchmark.py` scores model choices against MUSDB; see
[evaluation.md](evaluation.md).
