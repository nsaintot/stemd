#!/bin/bash
# Assemble stemd.app.
#
# One thing makes this more than a copy: weights are fetched on first run into
# Application Support, so the bundle stays small. STEMD_EMBED_MODELS puts them
# in Contents/Resources for an offline install, which is also why --models
# resolves relative to the exe.
#
# There is nothing to vendor. MLX links statically and its Metal shaders are
# embedded in the binary, so the executable depends on system frameworks alone,
# no Frameworks directory, no rpath surgery, no venv. It used to carry 223 MB of
# libtorch dylibs that had to be signed inside-out before the bundle could be.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-$ROOT/dist/stemd.app}"
MODELS="$ROOT/models"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"

# Only the embed/link paths need local weights; the default bundle fetches on
# first run and does not care whether models/ exists.
if [ "${STEMD_LINK_MODELS:-0}" = "1" ] || [ "${STEMD_EMBED_MODELS:-0}" = "1" ]; then
  if [ ! -f "$MODELS/htdemucs.safetensors" ]; then
    echo "no weights in $MODELS — let the server fetch them once, or copy them in" >&2
    exit 1
  fi
fi

# The oldest macOS this bundle claims to run on, and it has to be said out loud
# in three places that must agree: here for the Rust side's `minos`, in
# mlx-sys's build.rs for the Metal shaders, and in LSMinimumSystemVersion below.
# Unset, each of them picks up the SDK of whatever machine ran this script, and
# the result launches on an older Mac and then fails the first time it touches
# the GPU. See MACOS_DEPLOYMENT_TARGET in vendor/mlx-rs-stemd/mlx-sys/build.rs.
MACOS_MIN="14.0"

# MLX compiles its Metal kernels at build time, so `metal` has to be reachable
# or the build stops part way through with a compiler error per kernel.
#
# Found rather than named, because where it lives moved. It used to come with
# the Command Line Tools; it is now a separate component installed under an
# Xcode and mounted from a MobileAsset cryptex, so a machine can have Xcode, a
# current SDK and no shader compiler at all. The selection in effect is tried
# first, so anyone whose xcode-select already points at a full Xcode never
# enters the loop and nothing here overrides a deliberate choice.
ensure_metal() {
  if xcrun --find metal >/dev/null 2>&1; then
    return 0
  fi
  for candidate in /Applications/Xcode*.app/Contents/Developer; do
    [ -d "$candidate" ] || continue
    if DEVELOPER_DIR="$candidate" xcrun --find metal >/dev/null 2>&1; then
      export DEVELOPER_DIR="$candidate"
      echo "  metal: via $candidate"
      return 0
    fi
  done
  cat >&2 <<'NOMETAL'
no Metal shader compiler found, and MLX cannot be built without one.

`metal` no longer ships with the Command Line Tools. Install it under an Xcode:

  xcodebuild -downloadComponent MetalToolchain

then either point xcode-select at that Xcode or leave it under /Applications,
where this script looks.
NOMETAL
  return 1
}
ensure_metal

echo "building release binary (macOS $MACOS_MIN and up)..."
MACOSX_DEPLOYMENT_TARGET="$MACOS_MIN" \
  cargo build --release -p stemd-server --manifest-path "$ROOT/Cargo.toml"

echo "assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$ROOT/target/release/stemd-server" "$APP/Contents/MacOS/stemd"

# Weights are fetched on first run into Application Support, so the bundle does
# not carry 170 MB that never changes. STEMD_LINK_MODELS symlinks a local copy
# for development; STEMD_EMBED_MODELS copies it in for an offline install.
if [ "${STEMD_LINK_MODELS:-0}" = "1" ]; then
  ln -s "$MODELS" "$APP/Contents/Resources/models"
  echo "  models: symlinked (dev)"
elif [ "${STEMD_EMBED_MODELS:-0}" = "1" ]; then
  cp -R "$MODELS" "$APP/Contents/Resources/models"
  echo "  models: embedded ($(du -sh "$MODELS" | cut -f1))"
else
  echo "  models: fetched on first run"
fi

# The icon. `sips` and `iconutil` ship with macOS, so this needs nothing
# installed: the arithmetic that is not obvious (Apple's 824-of-1024 inset)
# was done once by scripts/make-icons.sh and lives in the committed master.
ICON_MASTER="$ROOT/resources/stemd-icon-mac-1024.png"
if [ -f "$ICON_MASTER" ]; then
  ICONSET="$(mktemp -d)/stemd.iconset"
  mkdir -p "$ICONSET"
  # Every slot macOS asks for. Finder picks by size, and a missing one is not
  # an error -- it is a blurry icon somewhere you did not look.
  for spec in 16:16x16 32:16x16@2x 32:32x32 64:32x32@2x \
              128:128x128 256:128x128@2x 256:256x256 512:256x256@2x \
              512:512x512 1024:512x512@2x; do
    px="${spec%%:*}"
    name="${spec#*:}"
    sips -z "$px" "$px" "$ICON_MASTER" --out "$ICONSET/icon_$name.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/stemd.icns"
  echo "  icon: stemd.icns ($(du -h "$APP/Contents/Resources/stemd.icns" | cut -f1))"
  ICON_PLIST='    <key>CFBundleIconFile</key>              <string>stemd</string>'
else
  echo "  icon: none ($ICON_MASTER is missing; run scripts/make-icons.sh)"
  ICON_PLIST=''
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>            <string>stemd</string>
$ICON_PLIST
    <key>CFBundleIdentifier</key>            <string>io.nasper.stemd</string>
    <key>CFBundleName</key>                  <string>stemd</string>
    <key>CFBundleDisplayName</key>           <string>stemd</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>$VERSION</string>
    <key>CFBundleVersion</key>               <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>        <string>$MACOS_MIN</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <!-- Advertises _stemd._tcp and serves on the LAN. -->
    <key>NSLocalNetworkUsageDescription</key>
    <string>stemd advertises itself so players on this network can find it.</string>
</dict>
</plist>
PLIST

# Nothing nested to sign inside-out any more: one executable, no dylibs.
#
# A Developer ID where the machine has one, ad-hoc otherwise. The difference is
# not cosmetic: an ad-hoc signature is trusted by the machine that made it and
# by nothing else, so that bundle opens here and is refused everywhere. Only the
# Developer ID one can be notarized, and only a notarized one opens on a Mac
# that has never seen this project.
#
# --options runtime turns on the hardened runtime, which notarization requires
# and which is therefore not optional even though nothing here needs relaxing.
# --timestamp so the signature keeps verifying after the certificate expires.
IDENTITY="${STEMD_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
  IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
fi
if [ -n "$IDENTITY" ]; then
  codesign --force --sign "$IDENTITY" --options runtime --timestamp "$APP"
  echo "  signed: $IDENTITY"
else
  codesign --force --sign - "$APP" 2>/dev/null \
    && echo "  signed: ad-hoc, so this opens on this machine and nowhere else" \
    || echo "  signed: FAILED (may still run locally)"
fi
codesign --verify --strict "$APP" && echo "  verified"

echo "done: $APP  ($(du -sh "$APP" | cut -f1))"
echo "run:  open '$APP'"
