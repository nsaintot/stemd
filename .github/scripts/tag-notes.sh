#!/usr/bin/env bash
# tag-notes.sh TAG - print the message of the annotated tag TAG, with LF line
# endings and without a tag signature. A lightweight tag or an empty message
# prints nothing.
#
# Read through the API because a checkout of a tag can lose the tag object.
set -euo pipefail
tag="$1"
: "${GH_REPO:?}"

ref=$(gh api "repos/$GH_REPO/git/ref/tags/$tag" --jq '"\(.object.type) \(.object.sha)"')
if [[ "${ref% *}" != tag ]]; then
    echo "::warning::$tag is a lightweight tag, so the update feed has no release notes" >&2
    exit 0
fi
gh api "repos/$GH_REPO/git/tags/${ref#* }" --jq .message \
    | tr -d '\r' | sed '/^-----BEGIN [A-Z]* SIGNATURE-----$/,$d'
