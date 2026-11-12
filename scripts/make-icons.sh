#!/bin/bash
# Derive the icon assets the build uses, from the two hand-made sources.
#
# Run when the artwork changes, not on every build: the outputs are committed
# so `bundle-app.sh` needs nothing but `sips` and `iconutil`, both of which are
# part of macOS. This script needs ImageMagick, which is not.
#
# Sources, both 2048 square:
#   stemd-icon-mac.png  finished macOS icon: 1648 squircle body (824 pt at 2x,
#                       Apple's grid), transparent margin and drop shadow
#   stemd-icon-win.png  full bleed with opaque corners; Windows rounds nothing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$ROOT/resources"

command -v magick >/dev/null || { echo "needs ImageMagick (brew install imagemagick)" >&2; exit 1; }

MAC_SRC="$RES/stemd-icon-mac.png"
WIN_SRC="$RES/stemd-icon-win.png"
for f in "$MAC_SRC" "$WIN_SRC"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

# 2048 -> 1024 master for the .icns.
magick "$MAC_SRC" -filter Lanczos -resize 1024x1024 \
  "$RES/stemd-icon-mac-1024.png"

# The window icon, which is what an unbundled run and every Windows build get.
# From the master so it has the same shape as the bundle icon.
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
