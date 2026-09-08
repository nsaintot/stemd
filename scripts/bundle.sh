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
VOLNAME="stemd $VERSION"

# The image's window is the artwork's. resources/stemd-dmg.png is the 2x
# rendition of a 640 by 400 window with two placeholder squares drawn on it,
# and the numbers below are where those squares are, as icon centres in
# window points. Move a square in the artwork, move its number here.
#
# Finder's window bounds are the frame, title bar included, so the content
# area only comes out the size of the artwork if the title bar is added on.
BACKGROUND="$ROOT/resources/stemd-dmg.png"
WINDOW_W=640
WINDOW_H=400
TITLE_BAR=28
ICON_SIZE=100
APP_X=149
APP_Y=173
APPS_X=491
APPS_Y=173
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
MOUNT=""
# An interrupted run must not leave the writable image mounted: the volume
# would sit on the desktop and hold the stage directory open.
cleanup() {
  [ -n "$MOUNT" ] && [ -d "$MOUNT" ] && hdiutil detach "$MOUNT" -quiet -force || true
  rm -rf "$STAGE"
}
trap cleanup EXIT
VOL="$STAGE/vol"
mkdir -p "$VOL/.background"
cp -R "$APP" "$VOL/"
ln -s /Applications "$VOL/Applications"

# Finder picks the rendition for the screen out of a TIFF that carries both,
# and the artwork is the 2x one, so the 1x is derived here rather than kept.
sips -Z "$WINDOW_W" "$BACKGROUND" --out "$STAGE/background-1x.png" >/dev/null
tiffutil -cathidpicheck "$STAGE/background-1x.png" "$BACKGROUND" \
  -out "$VOL/.background/stemd.tiff" 2>/dev/null

# Built writable, laid out, then compressed. The layout is a .DS_Store at the
# root of the volume, and Finder is the only thing that writes one, so the
# image is mounted and Finder is told what the window should look like.
#
# APFS rather than HFS+: on macOS 26 an HFS+ volume loses its .VolumeIcon.icns
# and the custom-icon flag when it is unmounted, so the image would come out
# with a plain disk icon. Every macOS the bundle runs on mounts APFS.
RW="$STAGE/rw.dmg"
hdiutil create -volname "$VOLNAME" -srcfolder "$VOL" -fs APFS -format UDRW \
  -ov -quiet "$RW"
MOUNT="$(hdiutil attach -readwrite -noverify -noautoopen "$RW" \
  | sed -n 's/.*\(\/Volumes\/.*\)$/\1/p')"
[ -d "$MOUNT" ] || { echo "the writable image did not mount" >&2; exit 1; }

osascript - "$VOLNAME" "$WINDOW_W" "$((WINDOW_H + TITLE_BAR))" "$ICON_SIZE" \
  "$APP_X" "$APP_Y" "$APPS_X" "$APPS_Y" <<'LAYOUT'
on run argv
  set volname to item 1 of argv
  set w to (item 2 of argv) as integer
  set h to (item 3 of argv) as integer
  set iconSize to (item 4 of argv) as integer
  set appX to (item 5 of argv) as integer
  set appY to (item 6 of argv) as integer
  set appsX to (item 7 of argv) as integer
  set appsY to (item 8 of argv) as integer
  tell application "Finder"
    tell disk volname
      open
      set current view of container window to icon view
      set toolbar visible of container window to false
      set statusbar visible of container window to false
      set bounds of container window to {200, 120, 200 + w, 120 + h}
      set opts to icon view options of container window
      set arrangement of opts to not arranged
      set icon size of opts to iconSize
      set text size of opts to 12
      set label position of opts to bottom
      set background picture of opts to file ".background:stemd.tiff"
      set position of item "stemd.app" of container window to {appX, appY}
      set position of item "Applications" of container window to {appsX, appsY}
      close
      open
      update without registering applications
      delay 1
      close
    end tell
  end tell
end run
LAYOUT

# The mounted volume wears the app's icon on the desktop and in the sidebar.
# After the layout, not before: Finder on macOS 26 deletes .VolumeIcon.icns
# and clears the custom-icon flag when it writes the window's .DS_Store, and
# an icon placed afterwards is left alone.
cp "$APP/Contents/Resources/stemd.icns" "$MOUNT/.VolumeIcon.icns"
SetFile -a C "$MOUNT"

sync
hdiutil detach "$MOUNT" -quiet

rm -f "$DMG"
hdiutil convert "$RW" -format UDZO -imagekey zlib-level=9 -quiet -o "$DMG"
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
