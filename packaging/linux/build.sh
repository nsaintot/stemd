#!/usr/bin/env bash
# Turn the staged tree into the three Linux packages for this machine's
# architecture.
#
#   build.sh [--version V] [--stage DIR] [--out DIR] [--only deb|rpm|appimage]
#
# Output: stemd-<version>-linux-<x86_64|aarch64>.{deb,rpm,AppImage}
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

version="$(awk '/^\[workspace.package\]/{f=1} f && /^version *=/{gsub(/"/,"");print $3;exit}' "$root/Cargo.toml")"
stage="$root/dist/linux"
out="$root/dist"
only=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) version="$2"; shift 2 ;;
        --stage) stage="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --only) only="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -x "$stage/opt/stemd/stemd" ]] || {
    echo "ERROR: $stage is not a staged tree. Run packaging/linux/stage.sh." >&2
    exit 1
}
mkdir -p "$out"
stage="$(cd "$stage" && pwd)"
out="$(cd "$out" && pwd)"

case "$(uname -m)" in
    x86_64) arch=amd64; uname_arch=x86_64 ;;
    aarch64) arch=arm64; uname_arch=aarch64 ;;
    *) echo "ERROR: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
base="stemd-${version}-linux-${uname_arch}"

want() { [[ -z "$only" || "$only" == "$1" ]]; }

if want deb || want rpm; then
    recipe="$(mktemp --suffix=.yaml)"
    trap 'rm -f "$recipe"' EXIT
    VERSION="$version" ARCH="$arch" envsubst '${VERSION} ${ARCH}' \
        < "$root/packaging/linux/nfpm.yaml.in" > "$recipe"
    for format in deb rpm; do
        want "$format" || continue
        echo "==> $format"
        (cd "$stage" && nfpm package --config "$recipe" --packager "$format" \
            --target "$out/$base.$format")
    done
fi

if want appimage; then
    echo "==> AppImage"
    appdir="$out/stemd.AppDir"
    rm -rf "$appdir"
    mkdir -p "$appdir"
    cp -a "$stage/opt" "$stage/usr" "$appdir/"
    install -m 0755 "$root/packaging/linux/AppRun" "$appdir/AppRun"
    install -m 0644 "$root/packaging/linux/stemd.desktop" "$appdir/stemd.desktop"
    cp "$appdir/usr/share/icons/hicolor/256x256/apps/stemd.png" "$appdir/stemd.png"
    ARCH="$uname_arch" appimagetool --no-appstream --comp zstd "$appdir" "$out/$base.AppImage"
    rm -rf "$appdir"
fi

for f in "$out/$base".{deb,rpm,AppImage}; do [[ -f "$f" ]] && du -h "$f"; done; true
