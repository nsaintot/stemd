# The client this was built for

Why the server ships two stems rather than four, and why the split lands where it
does. The constraints below are an embedded DJ player's; they are the reason for
several decisions that would otherwise look arbitrary.

This is the *client* side. What the server actually does is
[internals.md](internals.md); the contract between them is [api.md](api.md).

```
  Mac                                  Player
  ---                                  ------
  stemd                                sidecar daemon        player (JUCE app)
    |                                    |                     |
    |<--- mDNS _stemd._tcp ------------->|                     |
    |                                    |<-- track load hook --|  (LD_PRELOAD shim)
    |<--- POST /v1/jobs (raw PCM) -------|                     |
    |---- 2 stems: harmonics, vocals --->|                     |
    |                                  /dev/shm/stems-<hash>   |
    |                                    |--- mmap ----------->|
    |                                                          |
    |                                    drums = mix − harmonics − vocals
    |                                    gains applied, summed, THEN stretched
```

## Two rules that fall out of the measurements

**Mix before the time-stretcher, not after.** The audio RT thread owns one
isolated A72 and the player already runs at 104% CPU. Four independent
decode-and-stretch chains do not fit. All stems share an identical time base, so
apply gains, sum to one buffer, and feed the *existing* single stretcher. The
added RT cost is a few multiply-accumulate passes, which is also why the third
part is rebuilt inside that sum rather than materialised
([api.md](api.md#4-rebuild-the-third-part-while-mixing)).

**Nothing network-facing runs inside the player.** Discovery, HTTP and decode
live in a separate sidecar daemon. The `LD_PRELOAD` shim inside the player only
ever touches mapped memory.

## Why separation is offloaded at all

The player cannot do it. Measured on an RK3399 (4× Cortex-A53 @1.416 GHz + 2×
Cortex-A72 @1.608 GHz), Hybrid Demucs v3 costs **2.31 s of compute per 1 s of
audio** on the cores that remain after one A72 is fenced off for audio. The
aggregate is 7.52 GFLOP/s against a single M1 Firestorm core's 50.45; the gap to
real time is ~5x and the removable structure in the model is worth 1.5x. Detail
in [evaluation.md](evaluation.md#on-device-separation).

## Transfer and storage

- The player links at **1000 Mbit**, but `eth0` MTU is **1200**, not 1500: ~20%
  more per-packet overhead, and any UDP path must keep datagrams under ~1150.
- Two stems of a five-minute track: **53 MB** each as `pcm16`, about half that as
  FLAC. 16-bit is sufficient; the separation residual sits far above a −96 dB
  noise floor.
- Stems live in `/dev/shm` (tmpfs, 1.88 GB cap) against ~3.0 GB available RAM. An
  LRU of two or three tracks, current plus prefetched next, is the budget.
- Nothing is written to persistent storage. `/` is ramfs, unbounded and
  unreclaimable, so stems must never be written there.
- Store the two transferred stems only, losslessly, keyed on `model_id`: the
  reasoning is in [api.md](api.md#caching-on-the-client).

## Discovery

- **Primary: mDNS.** `avahi-daemon`, `avahi-dnsconfd` and `dbus-daemon` already
  run on the player, `libavahi-client.so.3` is in the rootfs, and macOS publishes
  Bonjour natively.
- **Override: a file on the USB stick.** `/media/usb/sda1/.stemd` with
  `host=addr:port`. Travels with the DJ, works on any unit, keeps zero state on
  the player.
- **Precedence:** USB file, then mDNS, then the feature is off.

A hardcoded IP is not viable: `avahi-autoipd` is present, so with no DHCP server
(the normal club setup, just a switch), the player self-assigns `169.254.x.x`,
renegotiated per power cycle.

## Still open

- **Deterministic selection when two hosts advertise**, and a UI affordance
  showing which is bound. Silently binding the wrong host presents as "stems are
  broken".
- **Progressive delivery**, so playback can start before the transfer finishes.
  At gigabit a 53 MB stem is a few seconds, so this is probably unnecessary.
- **Auth.** The scope is a local network; revisit if that changes.
- **A genuine two-head model** rather than folding a four-stem one. Only worth it
  once quality is settled, and [evaluation.md](evaluation.md) says the spare
  budget cannot currently be converted into quality anyway.
