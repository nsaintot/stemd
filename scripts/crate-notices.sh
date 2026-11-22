#!/usr/bin/env bash
# Regenerate packaging/THIRD-PARTY-CRATES, the licences of the Rust crates
# compiled into stemd-server and stemd-cli. Rerun after a dependency change and
# commit the result.
#
# Needs cargo-about at the version below:
#   cargo install cargo-about --locked --version 0.9.3 --features cli
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
want="0.9.3"

have="$(cargo about --version 2>/dev/null | awk '{print $2}')" || true
if [[ "$have" != "$want" ]]; then
    echo "cargo-about $want needed, found ${have:-none}." >&2
    echo "  cargo install cargo-about --locked --version $want --features cli" >&2
    exit 1
fi

cargo about generate --locked --fail --workspace \
    --manifest-path "$root/Cargo.toml" \
    --config "$root/packaging/about.toml" \
    --output-file "$root/packaging/THIRD-PARTY-CRATES" \
    "$root/packaging/about.hbs"
echo "wrote packaging/THIRD-PARTY-CRATES"
