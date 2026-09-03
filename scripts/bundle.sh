#!/bin/bash
# Wrap stemd.app into a signed, notarized, stapled .dmg.
#
# Needs a keychain profile, made once:
#   xcrun notarytool store-credentials stemd-notary \
#       --apple-id <apple id> --team-id <team id>
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

# Two v's: `codesign -dv` does not print Authority. Full grep, not -q: SIGPIPE
# fails the pipeline under pipefail.
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

# `history` hits the network as well as the keychain, so only the error naming
# the keychain item means the profile is absent.
if ! NOTARY_CHECK="$(xcrun notarytool history --keychain-profile "$PROFILE" 2>&1)"; then
  if printf '%s' "$NOTARY_CHECK" | grep -q "No Keychain password item"; then
    echo "no notarytool profile named '$PROFILE'. Make one with" >&2
    echo >&2
    echo "  xcrun notarytool store-credentials $PROFILE \\" >&2
    echo "      --apple-id <your apple id> --team-id <your team id>" >&2
  else
    echo "notarytool could not use profile '$PROFILE'. This is not proof it is" >&2
    echo "missing, so do not run store-credentials. It said:" >&2
    echo >&2
    printf '%s\n' "$NOTARY_CHECK" | sed 's/^/  /' >&2
  fi
  exit 1
fi

# The app is notarized and stapled on its own, so the ticket is on what gets
# dragged out of the image. notarytool takes no bundles, hence the zip.
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
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create -volname "stemd $VERSION" -srcfolder "$STAGE" \
  -fs HFS+ -format UDZO -ov -quiet "$DMG"
say "$(du -h "$DMG" | cut -f1)"

# A ticket can only be stapled to something signed.
codesign --force --sign "$IDENTITY" --timestamp "$DMG"
say "signed: $IDENTITY"

echo "notarizing the image"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait

xcrun stapler staple "$DMG"
say "image stapled"

xcrun stapler validate "$DMG" >/dev/null && say "image ticket: valid"
xcrun stapler validate "$APP" >/dev/null && say "app ticket:   valid"
spctl -a -t open --context context:primary-signature -vv "$DMG"

echo
echo "done: $DMG"
echo "  signed, notarized and stapled: opens on a Mac that has never seen this"
