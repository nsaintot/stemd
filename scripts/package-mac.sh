#!/bin/bash
# Turn stemd.app into a disk image somebody else's Mac will open.
#
# `bundle-app.sh` builds and signs; this wraps, notarizes and staples. Three
# steps rather than one because each answers a different refusal:
#
#   signing        Gatekeeper knows who made it
#   notarization   Apple has scanned it and says so
#   stapling       the ticket travels with the file, so a Mac offline at first
#                  launch does not have to ask
#
# Skip the last two and the download opens with "Apple could not verify stemd is
# free of malware", with no button that says open anyway.
#
# Twice, because a ticket is stapled to a file and the app leaves the image. The
# app is notarized on its own and stapled first, then the image is built around
# the stapled copy and notarized and stapled itself. Do only the image and it
# verifies perfectly right up until somebody drags the app to Applications and
# launches it with no network, which is the one case stapling exists for. The
# second submission is the same code and Apple answers it from cache.
#
# Notarization needs credentials, which live in a keychain profile made once:
#
#   xcrun notarytool store-credentials stemd-notary \
#       --apple-id <your apple id> --team-id <your team id>
#
# That prompts for an app-specific password from appleid.apple.com. Nothing here
# reads it: notarytool takes the profile by name and the keychain keeps the rest.
# Without a profile this still builds and signs a disk image and says what is
# missing, so an unnotarized DMG is a deliberate act rather than an accident.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
APP="$DIST/stemd.app"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
DMG="$DIST/stemd-$VERSION-macos-arm64.dmg"
PROFILE="${STEMD_NOTARY_PROFILE:-stemd-notary}"

say() { printf '  %s\n' "$*"; }

if [ "${1:-}" != "--no-build" ]; then
  "$ROOT/scripts/bundle-app.sh" "$APP"
fi
[ -d "$APP" ] || { echo "no $APP; run scripts/bundle-app.sh first" >&2; exit 1; }

IDENTITY="${STEMD_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
  IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
fi

# An ad-hoc bundle cannot be notarized and must not be shipped, so it is caught
# here rather than three minutes later by Apple.
#
# Two traps in one line, both of which make this refuse a bundle that is fine.
# `codesign -dv` does not print Authority at all: that starts at the second v.
# And `grep -q` exits on the first match, which hands codesign a SIGPIPE, which
# under `pipefail` is a failed pipeline. So: two v's, and a grep that reads to
# the end.
AUTHORITY="$(codesign -dvv "$APP" 2>&1 | grep '^Authority=' | head -1 || true)"
case "$AUTHORITY" in
  "Authority=Developer ID Application"*) ;;
  *)
    echo "$APP is signed '${AUTHORITY#Authority=}', not with a Developer ID, so it" >&2
    echo "would be refused on every Mac but this one. Install the certificate and" >&2
    echo "run bundle-app.sh again." >&2
    exit 1
    ;;
esac

if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
  echo "no notarytool profile named '$PROFILE'. Nothing here can be notarized," >&2
  echo "and an unnotarized build is refused on every Mac but this one. Make one" >&2
  echo "with" >&2
  echo >&2
  echo "  xcrun notarytool store-credentials $PROFILE \\" >&2
  echo "      --apple-id <your apple id> --team-id <your team id>" >&2
  echo >&2
  echo "and run this again." >&2
  exit 1
fi

# The app first, on its own, so the ticket is on the thing that gets dragged out
# of the image. notarytool takes a zip, an image or an installer package, and a
# bundle is none of those, so it travels inside one for the round trip.
echo "notarizing the app, which takes a few minutes"
APPZIP="$DIST/stemd-app-for-notarization.zip"
rm -f "$APPZIP"
ditto -c -k --keepParent "$APP" "$APPZIP"
xcrun notarytool submit "$APPZIP" --keychain-profile "$PROFILE" --wait
rm -f "$APPZIP"
xcrun stapler staple "$APP"
say "app stapled"

echo "assembling $DMG"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
# The drag target. A window with the app and nothing to drop it on is the one
# thing every Mac user knows how to do and cannot.
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create -volname "stemd $VERSION" -srcfolder "$STAGE" \
  -fs HFS+ -format UDZO -ov -quiet "$DMG"
say "$(du -h "$DMG" | cut -f1)"

# The image is signed too. Notarization staples its ticket to the DMG, and a
# ticket can only be stapled to something signed.
codesign --force --sign "$IDENTITY" --timestamp "$DMG"
say "signed: $IDENTITY"

echo "notarizing the image"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait

xcrun stapler staple "$DMG"
say "image stapled"

# What a downloader's Mac will decide, asked the same way it asks. Both, since
# both are things somebody ends up double-clicking.
xcrun stapler validate "$DMG" >/dev/null && say "image ticket: valid"
xcrun stapler validate "$APP" >/dev/null && say "app ticket:   valid"
spctl -a -t open --context context:primary-signature -vv "$DMG"

echo
echo "done: $DMG"
echo "  signed, notarized and stapled: opens on a Mac that has never seen this"
