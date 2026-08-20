#!/bin/bash
# Package stemd for Debian and Ubuntu, depending on the CUDA stack instead of
# carrying it.
#
# The counterpart to bundle-linux.sh and the opposite trade. That script makes a
# 1.6 GB directory that runs on any machine with a driver, because it brings its
# own CUDA runtime, its own cuDNN and 4248 headers. This one makes about 30 MB
# and asks apt for the rest, which is what a package manager is for: one copy of
# cuDNN on the machine, upgraded when the machine is, rather than a second copy
# that can disagree with the first. That disagreement has already cost this port
# a day, when a 9.0 dispatcher drove 9.14 engines, loaded cleanly, and then
# found no execution plan for any graph.
#
# The tarball is still the right shape for a live USB, where there is no package
# manager and nothing persists. It is the wrong shape for everything else.
#
# Usage:  scripts/package-deb.sh [output-dir] [path-to-stemd-server]
#
# The binary has to exist already. Building it is bundle-linux.sh's job, or
# cargo's, and both need the CUDA environment that this script has no business
# guessing at a second time.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist}"
binary="${2:-$root/target/release/stemd-server}"

say() { printf '  %s\n' "$*"; }
die() { printf '\nstopped: %s\n' "$*" >&2; exit 1; }

# Where the CUDA packages put the two header trees MLX reads at run time. Both
# are under the `cuda` alternative rather than a versioned directory, so an
# upgrade from 13.3 to 13.4 moves the target and the package keeps working.
CUDA_PREFIX=/usr/local/cuda
CCCL_DIR="$CUDA_PREFIX/targets/x86_64-linux/include/cccl"

# Where the real binary goes. Not /usr/bin, and the reason is not tidiness.
#
# MLX locates the CCCL headers it hands NVRTC at `<binary dir>/../include/cccl`,
# and it learns the binary directory from dladdr, which reports argv[0] and does
# not resolve it. Measured on this machine: invoked by absolute path it reports
# that path, invoked through a symlink it reports the symlink, and invoked as a
# bare name found on PATH it reports `stemd-server` with no directory at all. In
# that last case the lookup becomes the relative path `include/cccl` and lands
# wherever the working directory happens to be.
#
# So the install has to own a directory whose shape MLX expects, and the thing
# on PATH has to exec into it by absolute path. See the wrapper below.
LIBDIR=/usr/lib/stemd

[ -f "$binary" ] || die "no binary at $binary. Build one first:
            scripts/bundle-linux.sh
        or, with the CUDA environment already set,
            cargo build --release -p stemd-server"

# Check it is the CUDA build and not a CPU-only one, by asking the binary rather
# than the environment that made it. A CPU-only MLX links, starts, reports
# `device: cpu`, separates correctly and runs about 190x slower, and nothing in
# the build says anything is wrong. Shipping one as a .deb would be shipping
# that silence to other people's machines.
if command -v objdump >/dev/null; then
    needed="$(objdump -p "$binary" | awk '/NEEDED/ {print $2}')"
    for lib in libcudart.so.13 libnvrtc.so.13 libcublasLt.so.13; do
        grep -qx "$lib" <<<"$needed" ||
            die "$binary does not link $lib, so it was built against a CPU-only
            MLX. Set MLX_BUILD_CUDA=1 and CUDNN_PATH, delete
            target/release/build/mlx-sys-*, and build again."
    done
    say "cuda: linked (libcudart, libnvrtc, libcublasLt)"
else
    say "cuda: unverified, no objdump on this machine"
fi

version="$(awk '/^\[workspace.package\]/{f=1} f && /^version *=/{gsub(/[",]/,"");print $3;exit}' "$root/Cargo.toml")"
[ -n "$version" ] || die "no version in $root/Cargo.toml"
arch="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
pkgdir="$out/stemd_${version}_${arch}"

rm -rf "$pkgdir"
mkdir -p "$pkgdir/DEBIAN" \
         "$pkgdir$LIBDIR/bin" \
         "$pkgdir$LIBDIR/include" \
         "$pkgdir/usr/bin" \
         "$pkgdir/usr/share/applications" \
         "$pkgdir/usr/share/icons/hicolor/256x256/apps" \
         "$pkgdir/usr/share/doc/stemd"

install -m 0755 "$binary" "$pkgdir$LIBDIR/bin/stemd-server"
[ -f "$root/target/release/stemd-cli" ] &&
    install -m 0755 "$root/target/release/stemd-cli" "$pkgdir$LIBDIR/bin/stemd-cli"

# A symlink rather than a copy of the headers: they belong to cccl-13-3, which
# is in Depends, and two copies of a header tree is the same mistake as two
# copies of cuDNN in a smaller size.
ln -s "$CCCL_DIR" "$pkgdir$LIBDIR/include/cccl"

# The wrapper is load-bearing, not a convenience. See the note on LIBDIR: exec
# by absolute path is what makes MLX's argv[0]-derived header lookup land in
# this package instead of in the user's working directory.
cat > "$pkgdir/usr/bin/stemd-server" <<EOF
#!/bin/sh
# MLX reads its own install directory from argv[0], so the path used here is
# what decides where it looks for the CCCL headers it hands NVRTC. Exec by
# absolute path, never by name.
exec $LIBDIR/bin/stemd-server "\$@"
EOF
chmod 0755 "$pkgdir/usr/bin/stemd-server"

if [ -f "$pkgdir$LIBDIR/bin/stemd-cli" ]; then
    cat > "$pkgdir/usr/bin/stemd-cli" <<EOF
#!/bin/sh
exec $LIBDIR/bin/stemd-cli "\$@"
EOF
    chmod 0755 "$pkgdir/usr/bin/stemd-cli"
fi

install -m 0644 "$root/resources/stemd-icon-win-256.png" \
    "$pkgdir/usr/share/icons/hicolor/256x256/apps/stemd.png"

# Absolute Exec for the same reason the wrapper exists. A desktop file that said
# `Exec=stemd-server` would hand MLX an argv[0] with no directory in it.
cat > "$pkgdir/usr/share/applications/stemd.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=stemd
GenericName=Stem separation
Comment=Separate a track into stems on the GPU
Exec=/usr/bin/stemd-server
Icon=stemd
Terminal=false
Categories=AudioVideo;Audio;
Keywords=stems;separation;demucs;audio;
StartupNotify=true
EOF

# Dependencies.
#
# Two sets, because two mechanisms put libraries into this process. dpkg-shlibdeps
# reads the ELF and finds everything the linker recorded, which is the CUDA
# libraries, BLAS and the C++ runtime. It cannot see the rest: winit dlopens X11
# through x11-dl and glutin dlopens GL, so neither appears in the ELF and both
# are absolutely required for the window to open. A package that trusted
# shlibdeps alone would install cleanly and then fail to start on a machine that
# happened to lack libxcursor.
#
# The list below is not guesswork. It is every object the running process had
# mapped, read from /proc/<pid>/maps on a working machine and mapped back to
# packages with dpkg -S.
dlopened="libgl1, libglx0, libglvnd0,
 libx11-6, libx11-xcb1, libxcb1, libxcb-dri3-0, libxcb-glx0, libxcb-present0,
 libxcb-randr0, libxcb-sync1, libxcb-xfixes0, libxcb-xkb1,
 libxcursor1, libxext6, libxfixes3, libxi6, libxrender1,
 libxkbcommon0, libxkbcommon-x11-0, libdrm2"

# The Bonjour compatibility layer, which discovery.rs calls through DNSServiceRegister.
service="libavahi-compat-libdnssd1, libavahi-client3, libavahi-common3, libdbus-1-3, libsystemd0"

# CUDA, from NVIDIA's apt repository.
#
# cuda-cudart-dev-13-3 is here for its headers, not for development: MLX
# compiles kernels with NVRTC while it runs and includes the CUDA runtime
# headers from $CUDA_PREFIX/include as it goes. Without them every separation
# dies at the first gather, on a server that started fine and said device: gpu.
#
# cuDNN is pinned at 9.5 because the cudnn-frontend version MLX builds against
# calls cudnnBackendPopulateCudaGraph, which cuDNN added in 9.5. Debian's own
# nvidia-cudnn is 9.0.0 and declares nvidia-cuda-dev (<< 13~), so it is both too
# old and for the wrong CUDA. It is deliberately not offered as an alternative.
cuda="cuda-cudart-13-3, cuda-cudart-dev-13-3, cccl-13-3,
 libcublas-13-3, libcufft-13-3, cuda-nvrtc-13-3,
 libcudnn9-cuda-13 (>= 9.5)"

base="libc6, libgcc-s1, libstdc++6, libgfortran5, zlib1g, libcap2, libcuda1"
blas="libopenblas0 | libblas3"

if command -v dpkg-shlibdeps >/dev/null 2>&1; then
    # Two different problems, and they need two different flags.
    #
    # --ignore-missing-info is for a library that is found but belongs to no
    # package, which is cuDNN on Debian 13 today: NVIDIA publish none for trixie
    # yet, so there is nothing for shlibdeps to name.
    #
    # -l is for a library that is not found at all, which is the same cuDNN when
    # it was installed by hand outside the linker path. Without it shlibdeps
    # stops at "cannot find library libcudnn.so.9" before it reports anything,
    # and --ignore-missing-info does not cover that case.
    search=()
    [ -n "${CUDNN_PATH:-}" ] && [ -d "$CUDNN_PATH/lib" ] && search+=(-l"$CUDNN_PATH/lib")
    IFS=: read -ra extra <<<"${LD_LIBRARY_PATH:-}"
    for d in "${extra[@]}"; do [ -n "$d" ] && [ -d "$d" ] && search+=(-l"$d"); done

    pushd "$pkgdir" >/dev/null
    mkdir -p debian && : > debian/control
    if found="$(dpkg-shlibdeps -O --ignore-missing-info "${search[@]}" \
                    "$pkgdir$LIBDIR/bin/stemd-server" 2>/dev/null)"; then
        base="${found#shlibs:Depends=}"
        say "shlibdeps: resolved from the ELF"
    else
        say "shlibdeps: failed, using the checked-in base list"
    fi
    rm -rf debian
    popd >/dev/null
else
    say "shlibdeps: not installed, using the checked-in base list"
fi

# Merge the two sets, drop repeats, and relax one constraint that shlibdeps gets
# wrong for a package that is meant to travel.
#
# It reads the driver's symbols file and emits libcuda1 (>= <version installed
# here>), which today is 610.57.04. That is a fact about this machine, not a
# requirement of stemd: the floor is CUDA 13's own, which is 580. Left alone the
# package would refuse to install on a working 590 desktop and say the driver
# was too old, which would not be true.
#
# Deduplication keys on the package name, so the versioned entry shlibdeps
# produced wins over the bare name in the lists above, and alternatives such as
# `libopenblas0 | libblas3` keep their first name as the key.
depends="$(printf '%s, %s, %s, %s, %s' "$base" "$blas" "$service" "$cuda" "$dlopened" \
    | tr ',' '\n' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/ /g' \
    | grep -v '^$' \
    | sed -E 's/^libcuda1( \(.*\))?$/libcuda1 (>= 580)/' \
    | awk '!seen[$1]++' \
    | paste -sd, - \
    | sed 's/,/, /g')"

# Say which glibc this build just committed to, because nothing in this script
# decides it: it is whatever the machine that ran cargo happens to have, and it
# is what decides who can install the result.
#
# Measured with apt on each target, using a control-only package carrying this
# exact Depends line. Built on Debian 13 it comes out as libc6 (>= 2.39) and
# libstdc++6 (>= 14), which Ubuntu 24.04 satisfies, and Ubuntu 22.04 (2.35,
# 12.3) and Debian 12 (2.36, 12.2) do not. Every other dependency here, CUDA
# and X11 included, resolves on all of them. So the way to reach the older
# targets is to build on the oldest one, not to edit this file.
floor="$(tr ',' '\n' <<<"$depends" | grep -E '^ *(libc6|libstdc\+\+6) ' | tr -s ' ' | sed 's/^ //' | paste -sd'; ' -)"
[ -n "$floor" ] && say "built against: $floor"

installed_kb="$(du -sk "$pkgdir" | cut -f1)"

cat > "$pkgdir/DEBIAN/control" <<EOF
Package: stemd
Version: $version
Architecture: $arch
Maintainer: nsaintot <neoternon@gmail.com>
Section: sound
Priority: optional
Homepage: https://github.com/nsaintot/stemd
Installed-Size: $installed_kb
Depends: $depends
Description: Stem separation service with a drop window
 stemd separates a track into stems on the GPU and serves the result over HTTP,
 announcing itself on the local network with mDNS so a player can find it.
 .
 It needs an NVIDIA card and a driver new enough for CUDA 13, which means 580 or
 later. libcuda1 is listed because the binary links libcuda.so.1 and will not
 start without it. On a machine that already runs a driver it is already
 satisfied; on one with no driver at all, apt will offer to install one, and
 that is a decision worth taking deliberately. See README.Debian.
EOF

# README.Debian, because the two things that can go wrong at install time are
# both about the machine rather than the package, and neither is obvious from a
# dependency error.
cat > "$pkgdir/usr/share/doc/stemd/README.Debian" <<'EOF'
stemd for Debian
================

The CUDA dependencies come from NVIDIA's apt repository, not from Debian. Add
it before installing:

  https://developer.download.nvidia.com/compute/cuda/repos/

Debian 13 (trixie) is currently a special case. NVIDIA publish cuDNN for
ubuntu2204, ubuntu2404 and debian12, but not yet for debian13, so
libcudnn9-cuda-13 cannot be satisfied there from any repository. Debian's own
nvidia-cudnn package does not help: it is 9.0.0, and it declares
nvidia-cuda-dev (<< 13~), so it is both older than the 9.5 this needs and built
for the wrong CUDA.

Until NVIDIA publish for trixie, install cuDNN 9.5 or later by hand, tell the
dynamic linker where it went, and install this package without that one
dependency:

  echo /opt/cudnn13/lib | sudo tee /etc/ld.so.conf.d/cudnn.conf
  sudo ldconfig
  sudo dpkg -i --force-depends stemd_*.deb

The ld.so.conf.d entry is the part that is easy to miss. Without it the package
installs, the window opens, the server reports device: gpu, and then every
separation fails on libcudnn.so.9 at run time. Everything else in Depends
resolves normally on trixie; cuDNN is the only gap.

Which distributions can install this depends on where it was built, not on
this package. A build made on Debian 13 requires glibc 2.39 and libstdc++ 14,
which Debian 13 and Ubuntu 24.04 have and Ubuntu 22.04 and Debian 12 do not.
Everything else in Depends resolves on all four. Build on the oldest target you
mean to support.

The driver is not a dependency. CUDA 13 needs 580 or later; check with
nvidia-smi. A machine whose driver came from NVIDIA's .run installer has
libcuda.so.1 without the libcuda1 package, and apt will want to install the
package anyway. Use the distribution's driver packages, or the ones from
NVIDIA's repository, rather than the .run installer.
EOF

# Debian wants the full text of a licence that is not in common-licenses, which
# is why MIT is inlined and the other two are references. LAME has a stanza of
# its own because it is compiled into the binary rather than depended on: the
# package ships it, so the package has to say so.
{
    cat <<'EOF'
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: stemd
Source: https://github.com/nsaintot/stemd

Files: *
Copyright: 2026 Nicolas Saintot
License: MIT or Apache-2.0

Files: crates/stemd-core/data/mode1_*.bin
Copyright: measurements of third-party hardware, not authored code
License: MIT or Apache-2.0
 Filter coefficients recovered by measurement rather than written, so the grant
 above does not purport to cover them. They run one conversion, 44.1 to 96 kHz,
 and only when a request asks for it.

Files: (statically linked) lame-3.100
Copyright: 1998-2017 The LAME Project
License: LGPL-2
 The MP3 encoder is LAME 3.100, compiled into this binary by way of the
 mp3lame-encoder crate rather than linked against a system library. Its terms
 are the GNU Library General Public License version 2. Upstream source:
 https://lame.sourceforge.io/

License: LGPL-2
 On Debian systems the full text is in /usr/share/common-licenses/LGPL-2.

License: Apache-2.0
 On Debian systems the full text is in /usr/share/common-licenses/Apache-2.0.

License: MIT
EOF
    sed 's/^/ /; s/^ $/ ./' "$root/LICENSE-MIT"
} > "$pkgdir/usr/share/doc/stemd/copyright"

printf 'stemd (%s) unstable; urgency=low\n\n  * Packaged from the tree at %s.\n\n -- nsaintot <neoternon@gmail.com>  %s\n' \
    "$version" "$version" "$(date -R)" \
    | gzip -9n > "$pkgdir/usr/share/doc/stemd/changelog.Debian.gz"

# Normalise modes. The heredocs above inherit the builder's umask, and a umask
# of 002 puts group-writable files in the package, which is a policy violation
# that depends on who ran the script rather than on anything in the tree.
find "$pkgdir" -type d -exec chmod 0755 {} +
find "$pkgdir" -type f -perm -u+x -exec chmod 0755 {} +
find "$pkgdir" -type f ! -perm -u+x -exec chmod 0644 {} +
dpkg-deb --root-owner-group --build "$pkgdir" "$out/stemd_${version}_${arch}.deb" >/dev/null
rm -rf "$pkgdir"

deb="$out/stemd_${version}_${arch}.deb"
printf '\n'
say "packaged: $deb"
say "$(du -h "$deb" | cut -f1), against $(tr -cd , <<<"$depends" | wc -c) dependencies"
say "install with: sudo apt install $deb"
