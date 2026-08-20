# Internals

How the server is built. Nothing here is needed to use it: see
[api.md](api.md) for that.

## Cache

Stems live in `~/Library/Caches/stemd`, keyed by the uploaded samples plus
everything else that changes the answer: declared rate and channels, both
formats, the output rate, whether the derived part ships, and the model's
identity. A hit is provably the bytes a miss would have produced; a cache that
can return something *different* is worse than none. Re-tracing a preset changes
its digest and invalidates its entries by itself.

The model's identity is a `;`-separated `k=v` list, not just the artefact's
digest: `m=<digest>;r=<recipe>;p=f16;o=0.25`. The recipe covers a preset that
changes how it uses a file the file itself has not changed, and the last two are
`--full-precision` and `--overlap`, which change the samples without touching a
byte of the artefact. They were missing, which meant stems separated at f16 came
back unchanged from a server restarted with `--full-precision`: the one thing
the paragraph above says cannot happen. New fields go on the **end**, and every
field after `m=` draws its value from a closed set with no `;` in it, so the
split from the right stays unambiguous: `--demucs-model` is arbitrary operator
input and sits in `m=`, and a free-form field placed anywhere but first could be
spelled to look like another configuration.

Most of the value is not on disk. Two clients often ask for the same track within
seconds of each other, while the first separation is still running, so a request
matching a job already in flight **joins it** and both finish together. Without
that, the disk cache would rarely fire for concurrent requests: by the time there
is anything to hit, both separations have run.

The in-flight check runs *before* the cache lookup, which leaves no window,
because the worker publishes stems before it marks a job done. A shared job counts
its waiters, so one client cancelling does not take the separation from the other.

### Two rules bound the disk

1. **A separation nobody pulls is deleted** after `--unfetched-ttl` (300 s). Every
   stem fetched restarts the clock, so a client part-way through collecting is
   never cut off, and an entry counts as consumed only once *all* its stems have
   been read at least once.
2. **The store stays under `--cache-max-gb`** (4 GB ≈ 80 tracks as FLAC, 37 as
   `pcm16`) by
   dropping least-recently-used entries. The most recent entry is never rotated,
   so a track larger than the whole budget is not separated and discarded in the
   same breath.

Both run on one sweep at a fifth of the TTL: sweeping at the same period as the
deadline would let an abandoned separation sit for twice as long as configured.

Job handles are not a third policy: a job is dropped when the cache reaps the
stems it points at, so there is one lifecycle rather than two that can disagree.
Failures hold no stems and expire on the same clock.

The index is in memory and cleared at startup: whether an entry was ever pulled
is not recorded on disk, and rule 1 turns on exactly that. Inventing a
classification for files that cannot be classified is worse than paying for the
first separation again.

Entries are published by writing to `{key}.part/` and renaming, so a kill
mid-write cannot leave a directory that looks complete. Deleting an entry while a
client is streaming from it is safe: the handler opens the file first, and an
unlinked file stays readable through an open descriptor. A fetch that has not
opened yet gets a `410`.

## Queue

Separation is **serialised**. One track saturates the GPU, so a second concurrent
job would finish neither any sooner, it would just make both slower and the
progress reporting meaningless.

What a queue buys over a mutex is everything a client can see:

- **FIFO order**, so submission order is completion order
- **A live position**, computed at read time because positions shift every time
  the worker takes a job
- **Backpressure**: `429` past `--queue-depth` (16), rather than unbounded memory
- **Cancellation**, of the running job as well as the waiting ones

`DELETE` never discards stems already on disk, it drops the handle, not the work.

Cancelling the *running* job is what a deck skipping a track needs. Because
separation is serialised, an abandoned track otherwise holds the next one behind
a full separation whose stems nobody will fetch: the entire latency budget spent
on discarded work. One forward pass is opaque, but a track is not: the backend
polls the progress sink between segments, so the wait is one segment (well under
a second at the default preset) rather than one track.

The partial overlap-add is dropped rather than finished. Finishing would at least
leave the stems in the cache for a later cue, but that trades a certain cost now
against a benefit nobody may ever collect, and the live track is the urgent one.
A re-cue of a skipped track simply separates again.

A model switch is applied by the worker **between jobs**, because the old
separator holds GPU state a running forward pass is using. The worker also
*builds* it: MLX weights have to be allocated on the thread that will evaluate
them, so what crosses the thread boundary is a closure, never a loaded model.
The switch thread does the 320 MB download and then waits for the answer, which
is what keeps a failed switch reportable and stops `/v1/health` naming a model
the worker has not got yet.

## Discovery

The server advertises `_stemd._tcp` through the system Bonjour responder
(`DNSServiceRegister`, in libSystem, so it costs no dependency). A hardcoded
address is not viable on networks without DHCP, where both ends self-assign
link-local addresses that change per power cycle.

The advertisement is withdrawn on every exit path:

| exit | mechanism |
|---|---|
| window closed / Cmd-Q | `App::on_exit` |
| `SIGINT` / `SIGTERM` | signal task |
| normal return | `Drop` (backstop) |

`Drop` alone is not enough. On macOS a Cmd-Q terminates through `NSApplication`
without unwinding, and a signal does not unwind either, so destructors never run:
the record would linger until its TTL expired, which is minutes of clients
trying to reach a dead server. Withdrawal is idempotent, so the paths can overlap.

Interfaces are logged at startup: at `debug`, since they only mean anything once
discovery has gone wrong, so a failure on a multi-homed host is a one-line
diagnosis rather than a packet capture:

```
interfaces: en0=192.0.2.11 en7=198.51.100.4 bridge100=203.0.113.1
```

`--no-mdns` disables it; `--instance` sets the name. Advertisement failure is
logged and ignored: a client with the address configured out of band still works.

## Window

The window is the default because this ships as an `.app`, where stdout goes
nowhere and there would otherwise be no way to use it at all. It is a drop target
first: a track dropped on it becomes a job, and the stems land in
`<track>-stems/` beside the source file. Under that are the model and output
controls, the list of what has been separated and where it went, and the log.
Closing it stops the server.

A drop is **the same job** a `POST /v1/jobs` makes: same hash ingredients, same
store, same worker, so a drop and a deck asking for the same track join one
separation, and the same file twice is a cache hit. `drops.rs` is an adapter,
differing at two points: the audio comes from a file rather than a socket, and it
always asks for the derived part, because a folder of stems carries no mix to
rebuild it from.

The log view is a second *view* over the ring buffer `GET /v1/logs` serves, never
a second logging path. Because the GUI owns the main thread on macOS, the HTTP
runtime is driven explicitly in both modes rather than through `#[tokio::main]`.

Window bugs are testable without a window. `ui/probe.rs` runs egui headless:
`Context::run_ui` takes synthetic pointer input and returns the shapes that would
have been drawn: enough to ask whether hovering something grows a scroll bar, or
whether a full log fills its box. Both were real bugs, both "fixed" by eye first.
Each such test ships with one that reproduces the bug, because a sweep that finds
nothing is not a passing test.

## Logging

`error` is work that did not happen, `warn` is work that happened differently,
`info` is the record of what the server did to somebody's audio, `debug` is
mechanics: timings, bookkeeping, paths that did no work. A whole separation
costs two `info` lines.

The two sinks are filtered apart: the console takes `info` and up, which is what
a headless run should print, while the buffer behind the window and `/v1/logs`
keeps `debug` too, so the level dropdown has something to reveal and the detail
that explains a bug is already recorded rather than one restart away. `RUST_LOG`
overrides both at once.

## Settings

The model, output format and output sample rate persist to
`~/Library/Application Support/stemd/settings.json`. Every field is optional and
stored as the string the matching flag would take, so the file stays
hand-editable and one unreadable field costs that field rather than the file.

A flag is for one run, so the store keeps two copies: what the file says and
what this run is using. Disabling the pinned control is not enough on its own: a
write serialises the whole document, so changing the *rate* in the window would
otherwise carry a `--output-format` flag into the file beside it.

Format and rate are a pair, and a pair that cannot exist is reconciled rather
than attempted: `mp3` has no 96 kHz mode, and LAME does not refuse one it lacks:
it turns on its own resampler and spends eighteen seconds a stem producing a
48 kHz file nobody asked for. The pair can be made invalid four ways (two flags,
two menus, a hand-edited file, a file from a version that allowed it), which is
why the invariant lives here rather than at each control.

## Shutdown

One way out, however it was asked for. `shutdown::now` stops the worker
(cancelling the running job, which the separator notices between segments),
withdraws the mDNS advertisement, flushes the streams, and calls `_exit`.

`_exit` rather than a return, deliberately. This was written against libtorch,
whose MPS allocator tore down through C++ static destructors that aborted if the
model was running, so Cmd-Q mid-separation crashed on the way out. libtorch is
gone and MLX has not been seen to do this, but it is the same shape of hazard: a
statically linked C++ runtime holding a Metal device and an allocator in globals,
and skipping `atexit` costs a dying process nothing.

**Closing the window mid-separation asks first.** The exit above is correct but
silent, and a quality separation is four minutes: closing one minute in threw
that minute away with no way back. The close request is vetoed, a modal names
the running track, and only `Quit anyway` lets it through. An idle window closes
on the first click, there is nothing to lose, so there is nothing to ask.

`Cmd-Q` is not covered and cannot be. It is `[NSApp terminate:]`, which reaches
`applicationWillTerminate:` and then calls `exit()` as soon as the delegate
returns; there is no point in that path where a window can say "wait". `Cmd-Q`
mid-separation is still safe, it goes through `shutdown::now` like everything
else, cancels the job and does not crash. It just does not ask.

## Writing the stems

Both halves of the stage, converting to the output rate, then encoding, run a
thread per stem, at most three. Stems share nothing but the staging directory and
the GPU is finished by the time this runs, so serially the stage cost the sum of
three when it could cost the longest. Measured over a two-minute track, three
parts, against what the same work costs in sequence:

| format | rate | serial | concurrent |
|---|---|---|---|
| mp3 | 48 kHz | 7.37 s | **2.56 s** |
| mp3 | 44.1 kHz | 6.80 s | **2.32 s** |
| flac | 96 kHz | 3.10 s | **1.10 s** |
| flac | 44.1 kHz | 1.75 s | **0.63 s** |
| pcm32 | 96 kHz | 1.33 s | **0.61 s** |

WAV and the raw formats are a conversion and a write, 50 ms for three, so they
take the same path and have nothing to gain from it.

The conversion half is uneven, and the obvious guess is wrong in both
directions: 44.1 → 48 kHz gains 2.7x *despite* threading internally already,
while 44.1 → 96 kHz gains nothing at all, being bound by the 92 MB a stem it
writes rather than by arithmetic. Neither loses, so it is concurrent for every
rate rather than by case.

LAME runs at its recommended quality rather than its slowest. `q=0` encoded a
seven-minute stem in twenty seconds, three of those ran one after another, and
the writing stage cost more than the separation. Measured against `q=2`: the two
outputs differ by −47 dB, five decibels *below* the coding noise both of them add
anyway, and across the whole range from `q=0` to `q=7` the noise against the
source moves 0.9 dB: in the wrong direction for the slow end. At 320 kbps
constant there are enough bits that the quantisation search has nothing left to
find. Both benchmarks are `#[ignore]`d tests in `stemfmt.rs`, meant to be re-run
rather than believed.

`separation_secs` deliberately excludes this stage: it is the model's cost, and a
client reading it wants to know what the separation was worth rather than what
the encoder was set to. The `done` line reports the writing time beside it, which
it did not use to: sixty-five seconds of encoding appeared in no total anyone
saw.

## Model loading

`htdemucs` built layer by layer on MLX, in `stemd-mlx`. Not a traced graph: MLX
has no equivalent of TorchScript, so every layer is a place the port can differ
from the original, which is why each stage is nulled against the reference
implementation before the next is built on it. A whole track lands at -123.7 dB
on Metal.

The per-stage arrangement is what makes that number worth anything, and it is
also what makes it portable as a *method* rather than as a value. A null is two
implementations of the same arithmetic agreeing to whatever floor their
arithmetic shares; change the backend and the floor moves, without anything
being wrong. So the figure above is Metal's, `tests/model.rs` asserts the -60 dB
bar that any backend must clear, and a second backend re-derives its own numbers
from `spectral` upward rather than inheriting these.

There is no manifest beside the weights and there does not need to be. The
architecture is compiled in as `htdemucs::Config`, every layer checks the shape
of the tensor it pulls, and an artefact that is not this architecture fails at
load naming the tensor that disagreed. How many models a file holds is asked of
the tensor names rather than guessed from the filename.

**Which models run is the preset's business, not the file's.** `Fast` is one
model over the track. `Balanced` and `Quality` are two, each supplying the one
stem it is best at, with `harmonics` as the remainder: `stemd_core::hybrid`,
and evaluation.md for why that arrangement rather than the obvious one.

A fine-tuned set like `htdemucs_ft` therefore does not load as a plain model:
four checkpoints each produce a usable version of exactly one source, so taking
all four sources from one of them would ship four stems that look right and are
wrong. `MlxSeparator` refuses it and says so. There used to be a third answer:
run all four and combine them by a weight matrix, which with the identity
matrix `htdemucs_ft` ships is four forwards to keep one source from each. That
is gone with the preset that used it.

Three things are load-bearing:

- **MLX belongs to the thread that built it**, not merely to one thread at a
  time. Metal only trips on a second live command encoder; CUDA keeps its stream
  registry in thread-local storage, so weights allocated elsewhere fail their
  first `eval` with `There is no Stream(gpu, N) in current thread`. The queue
  worker constructs its own separator for that reason. Both constrain callers
  rather than the design.
- **The model runs at float16 by default**, the normalisations excepted, which
  keep their statistics in float32. Without that exception this does not separate
  slightly worse, it produces NaN: a group norm at one group sums a million
  squares. `--full-precision` opts out, at 1.3x the time.
- **Everything around the model stays float32**: the spectrogram and its
  inverse, the per-branch standardisation, and the overlap-add across segments.
  Those are a few per cent of a forward pass and they are where half precision
  would actually hurt.

Two mistakes worth recording, because both were correct and quietly slow. The
inverse transform walked its 336 frames, placing each into a freshly zeroed
whole-segment buffer: 40% of a forward pass, fixed by noticing that `n_fft` is
four hops, so frames four apart do not overlap and the whole overlap-add is four
reshapes. And the normalisations were written out by hand: eleven passes over
the tensor where `fast::layer_norm` is one fused kernel, sixty-four times a
forward pass. Together those took a 120 s track from 15.5 s to 6.5 s.

The upload is hashed and decoded on a blocking pool, not on the async runtime:
hashing a five-minute upload takes ~160 ms and decoding another ~30, either of
which would stall the stem downloads of a client already waiting.

## Model artefacts

Weights are 168–672 MB that never change, so they are fetched on first run from
[this project's release](https://github.com/nsaintot/stemd/releases/tag/models-v2)
into `~/Library/Application Support/stemd/models`: never into the bundle, whose
contents the code signature covers.

Mirrored rather than fetched from where they came from, which is a trade. The
demucs conversions are mlx-community's and the RoFormer one has no upstream at
all; either publisher could retag or vanish, and a pinned digest turns that into
a failed install rather than a wrong model, but a failed install is still a
broken app. One release under this project's control is one thing to keep alive.

`Quality` is the only preset needing two files, and its second is `Balanced`'s,
pinned identically. The fetcher skips what it finds, so a machine with Balanced
installed pays for the difference and not the pair.

Downloads are pinned by SHA-256 and verified before the file is put in place, via
a temporary file and a rename so an interrupted run cannot leave something
half-written that looks complete. There is deliberately no way to skip that check:
an unpinned digest would accept whatever the network returned, for a file whose
tensors are then loaded and multiplied by every sample of your audio.

**The cache is re-hashed on every load, not trusted by filename.** Verifying at
download time only covers the download; a file can rot afterwards, and the
shape checks catch a wrong artefact but not a bit-flipped one. Hashing costs
about a second against a process that then spends tens of seconds on a track.

A cached file that fails is deleted and refetched automatically. Under `--offline`
it is refused and left in place, since with no way to replace it, deleting only
destroys what you might want to look at. A mismatch outside the cache is reported
but not acted on: a locally converted artefact is *expected* to differ from the
published one.

Resolution order is `--models`, then alongside the executable (including
`Contents/Resources`), then the cache, then the network.

## Bundle

Two things a bundle needs that a `cargo run` does not:

- **Models are found relative to the executable.** A bundle launched from Finder
  starts with the working directory set to `/`, so a relative `--models` cannot
  resolve against the cwd.
- **`-psn_*` in argv is discarded.** Finder can pass a process serial number when
  launching a bundle, and clap would reject it as an unknown flag.

`bundle-app.sh` copies one executable in. There is nothing to vendor: MLX links
statically and embeds its Metal shaders, so the binary depends on system
frameworks alone: no `Contents/Frameworks`, no `install_name_tool`, no rpath.
The bundle is 29 MB where it used to be 223, and the 194 MB difference was five
libtorch dylibs that had to be signed inside-out before the bundle itself could
be. It is ad-hoc signed, which launches on the machine that built it and nowhere
else; distribution needs a Developer ID, the hardened runtime and notarization.
`Info.plist` carries `NSLocalNetworkUsageDescription` because the app advertises
over mDNS and serves on the LAN.

## Tools

`tools/` ships nothing and runs nothing at serve time, but it is tracked, because
it is how everything the null tests read was produced. `tools/eval/make_*.py`
dump reference tensors stage by stage: spectrogram, one encoder layer, one
decoder layer, the transformer, a whole segment, a whole track, and the tests in
`crates/stemd-mlx/tests/` compare against them in dB.

`pyproject.toml` still lists torch. Not to run anything: published model weights
are PyTorch files and stemd runs MLX, so adding a model means opening the
original once to convert its tensors, which is what `tools/export/` does, and
where the RoFormer artefact came from, nobody having published an MLX one.
`tools/eval/benchmark.py` scores model choices against MUSDB; see
[evaluation.md](evaluation.md).
