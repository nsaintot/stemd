#!/usr/bin/env bash
# release-draft.sh VERSION TAG DIR NOTES_MD - fill the draft release TAG from
# the packages in DIR and the release notes in NOTES_MD: write and sign
# appcast.xml, write SHA256SUMS.txt, set the release text and upload every file
# in DIR. The draft is created if it does not exist; a published release fails
# the run.
#
# The release text is the notes, a table of the downloads, and GitHub's
# generated list of pull requests in a collapsed block. When NOTES_MD is empty,
# that generated list is the notes, for the release text and the feed.
#
# With NOTES_ONLY=1, DIR holds only the .dmg and the draft's SHA256SUMS.txt,
# and only appcast.xml is uploaded again.
set -euo pipefail
version="$1"
tag="$2"
dir="$3"
notes_md="$4"
: "${GH_REPO:?}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

draft=$(gh release view "$tag" --json isDraft --jq .isDraft 2>/dev/null || true)
if [[ "$draft" == false ]]; then
    echo "::error::$tag is published; its text and files are no longer changed" >&2
    exit 1
fi

gh api "repos/$GH_REPO/releases/generate-notes" -f tag_name="$tag" --jq .body \
    | tr -d '\r' > "$work/generated.md"
if [[ -z "$(tr -d '[:space:]' < "$notes_md")" ]]; then
    notes_md="$work/generated.md"
fi

if [[ -n "${NOTES_ONLY:-}" ]]; then
    [[ -f "$dir/SHA256SUMS.txt" ]] || { echo "::error::$tag has no SHA256SUMS.txt" >&2; exit 1; }
else
    rm -f "$dir/SHA256SUMS.txt" "$dir/appcast.xml"
    (cd "$dir" && sha256sum -- * > "$work/SHA256SUMS.txt")
    mv "$work/SHA256SUMS.txt" "$dir/"
fi
cat "$dir/SHA256SUMS.txt"
packages=$(awk '{print $2}' "$dir/SHA256SUMS.txt")

# Sparkle's window shows the notes as HTML.
gh api markdown -f text="$(cat "$notes_md")" -f mode=gfm -f context="$GH_REPO" > "$work/notes.html"
"$script_dir/appcast.sh" "$version" "$tag" "$dir" "$work/notes.html"

# Prints one table row per host, with a link per package in SHA256SUMS.txt.
downloads() {
    local base="https://github.com/$GH_REPO/releases/download/$tag"
    local row host files file links
    echo "| Host | Package |"
    echo "|---|---|"
    for row in \
        "macOS 14 or later, Apple silicon|macos-arm64.dmg:dmg" \
        "Windows x64|setup.exe:installer windows-x64.zip:zip" \
        "Debian, Ubuntu|linux-x86_64.deb:x86_64 linux-aarch64.deb:aarch64" \
        "Fedora, RHEL|linux-x86_64.rpm:x86_64 linux-aarch64.rpm:aarch64" \
        "Any Linux distribution|linux-x86_64.AppImage:x86_64 linux-aarch64.AppImage:aarch64"; do
        host="${row%%|*}"
        files="${row#*|}"
        links=""
        for file in $files; do
            name="stemd-$version-${file%%:*}"
            grep -qxF "$name" <<< "$packages" || continue
            links+="${links:+ · }[${file#*:}]($base/$name)"
        done
        if [[ -n "$links" ]]; then
            echo "| $host | $links |"
        fi
    done
    echo
    echo "Windows and Linux use an NVIDIA GPU with driver 580 or later, else the CPU."
    echo "Linux needs glibc 2.35 or later. Checksums are in \`SHA256SUMS.txt\`."
}

{
    cat "$notes_md"
    echo
    echo "## Downloads"
    echo
    downloads
    if [[ "$notes_md" != "$work/generated.md" ]]; then
        echo
        echo "<details>"
        echo "<summary>Pull requests</summary>"
        echo
        cat "$work/generated.md"
        echo
        echo "</details>"
    fi
} > "$work/body.md"

if [[ -n "$draft" ]]; then
    gh release edit "$tag" --notes-file "$work/body.md" >/dev/null
else
    gh release create "$tag" --draft --verify-tag --title "Stemd $version" \
        --notes-file "$work/body.md" >/dev/null
fi
if [[ -n "${NOTES_ONLY:-}" ]]; then
    gh release upload "$tag" --clobber -- "$dir/appcast.xml"
else
    (cd "$dir" && gh release upload "$tag" --clobber -- *)
fi
