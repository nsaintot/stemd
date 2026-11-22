#!/bin/bash
# Assemble Stemd.app.
#
# Weights are downloaded on first run into Application Support, so the bundle
# stays small. STEMD_EMBED_MODELS puts them in Contents/Resources for an offline
# install, which is why --models resolves relative to the executable.
#
# Little needs vendoring. MLX links statically, so the executable depends only
# on system frameworks: no rpath changes, no venv. Two files travel with it:
# mlx.metallib, the compiled Metal kernels, and Sparkle.framework, the updater,
# which the app loads at run time. See below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-$ROOT/dist/Stemd.app}"
MODELS="$ROOT/models"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"

# Only the embed and link options need local weights; the default bundle
# downloads on first run and ignores whether models/ exists.
if [ "${STEMD_LINK_MODELS:-0}" = "1" ] || [ "${STEMD_EMBED_MODELS:-0}" = "1" ]; then
  if [ ! -f "$MODELS/htdemucs.safetensors" ]; then
    echo "no weights in $MODELS — let the server fetch them once, or copy them in" >&2
    exit 1
  fi
fi

# The oldest macOS this bundle supports. It must be stated in three places that
# agree: here for the executable's `minos`, in mlx-sys's build.rs for the Metal
# shaders, and in LSMinimumSystemVersion below. Unset, each picks up the SDK of
# the machine running this script, and the result launches on an older Mac and
# fails the first time it uses the GPU. See MACOS_DEPLOYMENT_TARGET in
# vendor/mlx-rs-stemd/mlx-sys/build.rs.
MACOS_MIN="14.0"
# Built with an explicit target so the flags below reach only what ships.
TARGET="aarch64-apple-darwin"

# MLX compiles its Metal kernels at build time, so `metal` must be reachable, or
# the build stops part way with a compiler error per kernel.
#
# The compiler is searched for because its location varies: it is a separate
# component installed under an Xcode and mounted from a MobileAsset cryptex, so
# a machine can have Xcode and a current SDK and no shader compiler. The active
# selection is tried first, so a machine whose xcode-select already points at a
# full Xcode skips the loop, and nothing here overrides a deliberate choice.
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

# Source paths go into the binary: every panic names its file, and so does the
# debug info. As built, they are this machine's absolute paths. The prefixes are
# rewritten to names that say what each file is and nothing about the disk it
# was built on.
# The source trees are remapped, never the whole checkout: a prefix covering
# target/ stops rustc from finding the proc-macro crates it just built there.
REMAP="--remap-path-prefix=$ROOT/crates=stemd/crates"
REMAP="$REMAP --remap-path-prefix=$ROOT/vendor=stemd/vendor"
REMAP="$REMAP --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo"
REMAP="$REMAP --remap-path-prefix=$(rustc --print sysroot)=rustc"

# The deployment floor goes to the linker as a flag. MACOSX_DEPLOYMENT_TARGET in
# the environment would reach every darwin artefact cargo builds, proc-macro
# dylibs included, and with this toolchain some of those come out with a
# LINKEDIT string pool that dyld refuses to load, which cargo reports as "can't
# find crate". Under --target, RUSTFLAGS does not apply to host artefacts, so
# the proc macros build normally and only the executable carries the floor.
echo "building release binary (macOS $MACOS_MIN and up)..."
RUSTFLAGS="${RUSTFLAGS:-} $REMAP -C link-arg=-mmacosx-version-min=$MACOS_MIN" \
  cargo build --release --target "$TARGET" -p stemd-server --manifest-path "$ROOT/Cargo.toml"

echo "assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$ROOT/target/$TARGET/release/stemd-server" "$APP/Contents/MacOS/stemd"

# The Metal kernels are not in the binary. MLX compiles them to mlx.metallib and
# loads that at run time, first from the directory holding its own code, which
# for a static link is the executable's directory, and then from the build
# tree's absolute path compiled into it. On the build machine the second path
# exists, so the app and every test work there; on any other Mac the first GPU
# array comes back empty and the separation panics on an out-of-range index.
#
# The path comes from the binary itself, never from a guess at cargo's output
# directory, so the shipped library is the one this executable was built
# against.
#
# The file lives in Resources, where codesign seals it as data; a non-code file
# directly in MacOS fails verification as unsigned nested code. MLX finds it
# through the symlink, and codesign seals symlinks as symlinks.
METALLIB="$(strings -n 8 "$APP/Contents/MacOS/stemd" | grep '/mlx\.metallib$' | head -1)"
if [ -z "$METALLIB" ] || [ ! -f "$METALLIB" ]; then
  echo "the binary names no mlx.metallib that exists (got '${METALLIB:-nothing}')." >&2
  echo "Without it the app only runs on this machine. Build with MLX_METAL_JIT=OFF" >&2
  echo "and check target/release/build/mlx-sys-*/out/build/lib/mlx.metallib." >&2
  exit 1
fi
cp "$METALLIB" "$APP/Contents/Resources/mlx.metallib"
ln -s ../Resources/mlx.metallib "$APP/Contents/MacOS/mlx.metallib"
echo "  metal kernels: mlx.metallib ($(du -h "$APP/Contents/Resources/mlx.metallib" | cut -f1))"

# The terms file of an artefact, from resources/model-terms. The mapping matches
# crates/stemd-server/src/models/terms.rs.
model_terms() {
  case "$1" in
    stemd_roformer) echo stemd_roformer ;;
    htdemucs|htdemucs_ft) echo demucs ;;
    bs_polarformer|bs_roformer_viperx) echo roformer-conversion ;;
    *) return 1 ;;
  esac
}

# Put <artefact>.LICENSE.txt beside each weights file in $1, as the server does
# after a download.
embed_model_terms() {
  local weights artefact terms
  for weights in "$1"/*.safetensors; do
    [ -f "$weights" ] || continue
    artefact="$(basename "$weights" .safetensors)"
    if ! terms="$(model_terms "$artefact")"; then
      echo "no terms for $artefact; add it to model_terms in $0" >&2
      exit 1
    fi
    cp "$ROOT/resources/model-terms/$terms.txt" "${weights%.safetensors}.LICENSE.txt"
  done
}

# Weights are fetched on first run into Application Support, so the bundle does
# not carry 170 MB that never changes. STEMD_LINK_MODELS symlinks a local copy
# for development; STEMD_EMBED_MODELS copies it in for an offline install.
if [ "${STEMD_LINK_MODELS:-0}" = "1" ]; then
  ln -s "$MODELS" "$APP/Contents/Resources/models"
  echo "  models: symlinked (dev)"
elif [ "${STEMD_EMBED_MODELS:-0}" = "1" ]; then
  cp -R "$MODELS" "$APP/Contents/Resources/models"
  embed_model_terms "$APP/Contents/Resources/models"
  echo "  models: embedded ($(du -sh "$MODELS" | cut -f1)), each with its terms"
else
  echo "  models: fetched on first run"
fi

# The licences and notices listed in packaging/notices.list.
notices=0
while read -r notice; do
  case "$notice" in ''|'#'*) continue ;; esac
  cp "$ROOT/$notice" "$APP/Contents/Resources/"
  notices=$((notices + 1))
done < "$ROOT/packaging/notices.list"
echo "  notices: $notices files"

# Sparkle, the updater. The app loads it from Contents/Frameworks; without it
# the app runs with no updater. Its XPC services are for sandboxed apps and are
# removed.
SPARKLE_VERSION="2.10.0"
SPARKLE_SHA256="c2bf58aa8387266ac179357b1415d6f2635f044da8be41042af32425dae6da0c"
SPARKLE_DIR="$ROOT/target/sparkle-$SPARKLE_VERSION"
if [ ! -d "$SPARKLE_DIR/Sparkle.framework" ]; then
  mkdir -p "$SPARKLE_DIR"
  archive="$SPARKLE_DIR/Sparkle.tar.xz"
  curl -fsSL -o "$archive" \
    "https://github.com/sparkle-project/Sparkle/releases/download/$SPARKLE_VERSION/Sparkle-$SPARKLE_VERSION.tar.xz"
  got="$(shasum -a 256 "$archive" | cut -d' ' -f1)"
  if [ "$got" != "$SPARKLE_SHA256" ]; then
    echo "Sparkle-$SPARKLE_VERSION.tar.xz has SHA-256 $got, expected $SPARKLE_SHA256" >&2
    rm -rf "$SPARKLE_DIR"
    exit 1
  fi
  tar -xf "$archive" -C "$SPARKLE_DIR" Sparkle.framework
  rm "$archive"
fi
SPARKLE="$APP/Contents/Frameworks/Sparkle.framework"
mkdir -p "$APP/Contents/Frameworks"
ditto "$SPARKLE_DIR/Sparkle.framework" "$SPARKLE"
rm -rf "$SPARKLE/Versions/B/XPCServices" "$SPARKLE/XPCServices"
SPARKLE_PUBLIC_KEY="$(tr -d '[:space:]' < "$ROOT/packaging/update-ed25519.pub")"
echo "  updater: Sparkle $SPARKLE_VERSION"

# The icon. `sips` and `iconutil` ship with macOS, so this needs nothing
# installed: the non-obvious arithmetic (Apple's 824-of-1024 inset) is done by
# scripts/make-icons.sh and lives in the committed master.
ICON_MASTER="$ROOT/resources/stemd-icon-mac-1024.png"
if [ -f "$ICON_MASTER" ]; then
  ICONSET="$(mktemp -d)/stemd.iconset"
  mkdir -p "$ICONSET"
  # Every slot macOS asks for. Finder picks by size, and a missing slot raises
  # no error; it shows up as a blurry icon somewhere.
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
    <key>CFBundleName</key>                  <string>Stemd</string>
    <key>CFBundleDisplayName</key>           <string>Stemd</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>$VERSION</string>
    <key>CFBundleVersion</key>               <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>        <string>$MACOS_MIN</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <!-- Sparkle: the feed of the latest release, the key its .dmg is signed with. -->
    <key>SUFeedURL</key>                     <string>https://github.com/nsaintot/stemd/releases/latest/download/appcast.xml</string>
    <key>SUPublicEDKey</key>                 <string>$SPARKLE_PUBLIC_KEY</string>
    <key>SUEnableAutomaticChecks</key>       <true/>
    <!-- Advertises _stemd._tcp and serves on the LAN. -->
    <key>NSLocalNetworkUsageDescription</key>
    <string>Stemd advertises itself so players on this network can find it.</string>
</dict>
</plist>
PLIST

# Sparkle is signed inside-out, helpers first, before the bundle that seals it.
# The metallib is sealed as a resource.
#
# The bundle is signed with a Developer ID when the machine has one, and ad-hoc
# otherwise. An ad-hoc signature is trusted only by the machine that made it, so
# that bundle opens here and is refused everywhere else. Only a Developer ID
# signature can be notarized, and only a notarized bundle opens on a Mac that
# has never seen this project.
#
# --options runtime turns on the hardened runtime, which notarization requires,
# even though nothing here needs relaxing. --timestamp keeps the signature
# verifying after the certificate expires.
IDENTITY="${STEMD_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
  IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
fi
if [ -n "$IDENTITY" ]; then
  for code in "$SPARKLE/Versions/B/Autoupdate" "$SPARKLE/Versions/B/Updater.app" "$SPARKLE"; do
    codesign --force --sign "$IDENTITY" --options runtime --timestamp "$code"
  done
  codesign --force --sign "$IDENTITY" --options runtime --timestamp "$APP"
  echo "  signed: $IDENTITY"
else
  for code in "$SPARKLE/Versions/B/Autoupdate" "$SPARKLE/Versions/B/Updater.app" "$SPARKLE"; do
    codesign --force --sign - "$code"
  done
  codesign --force --sign - "$APP" 2>/dev/null \
    && echo "  signed: ad-hoc, so this opens on this machine and nowhere else" \
    || echo "  signed: FAILED (may still run locally)"
fi
codesign --verify --strict "$APP" && echo "  verified"

echo "done: $APP  ($(du -sh "$APP" | cut -f1))"
echo "run:  open '$APP'"
