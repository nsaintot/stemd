#!/bin/bash
# Derive the icon assets the build uses, from the two hand-made sources.
#
# Run when the artwork changes, not on every build: the outputs are committed
# so `bundle-app.sh` needs nothing but `sips` and `iconutil`, both of which are
# part of macOS. This script needs ImageMagick, which is not.
#
# The one piece of arithmetic worth writing down is the macOS inset. Apple's
# icon grid puts the *body* of a standard app icon in 824 points of a 1024
# point canvas (80.47%) with the rest left transparent so neighbouring icons
# in the dock line up. `stemd-icon-mac.png` is 1648 square, which is exactly
# twice 824, so it is the body at 2x and not a finished icon: used as-is it
# would render about a quarter larger than every icon beside it. Centring it on
# a 2048 canvas restores the margin, and 2048 rather than 1024 so the downscale
# to the master happens once, from the full resolution.
#
# The Windows source is a full square with opaque corners, which is right for
# that platform: Windows applies no rounding of its own.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$ROOT/resources"

command -v magick >/dev/null || { echo "needs ImageMagick (brew install imagemagick)" >&2; exit 1; }

MAC_SRC="$RES/stemd-icon-mac.png"
WIN_SRC="$RES/stemd-icon-win.png"
for f in "$MAC_SRC" "$WIN_SRC"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

# 1648 body -> centred on 2048 transparent -> 1024 master for the .icns.
magick "$MAC_SRC" \
  -background none -gravity center -extent 2048x2048 \
  -filter Lanczos -resize 1024x1024 \
  "$RES/stemd-icon-mac-1024.png"

# The window icon, which is what an unbundled run and every Windows build get.
# From the padded master so it is the same shape as the bundle's.
magick "$RES/stemd-icon-mac-1024.png" -filter Lanczos -resize 256x256 \
  "$RES/stemd-icon-mac-256.png"
magick "$WIN_SRC" -filter Lanczos -resize 256x256 \
  "$RES/stemd-icon-win-256.png"

# The executable's own icon, which is what Explorer, the Start menu shortcut and
# Add/Remove Programs show. Seven sizes because Windows picks per context and
# scales the nearest one when the size it wants is absent; 256 is stored as PNG,
# which is what makes a modern .ico a reasonable size.
magick "$WIN_SRC" -filter Lanczos \
  -define icon:auto-resize=256,128,64,48,32,24,16 \
  "$RES/stemd.ico"

echo "wrote:"
for f in stemd-icon-mac-1024.png stemd-icon-mac-256.png stemd-icon-win-256.png stemd.ico; do
  printf '  %-26s %s\n' "$f" "$(magick identify -format '%wx%h %b' "$RES/$f[0]")"
done
