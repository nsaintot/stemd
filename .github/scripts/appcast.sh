#!/usr/bin/env bash
# appcast.sh VERSION TAG DIR NOTES_HTML - write DIR/appcast.xml, the Sparkle
# feed for the macOS .dmg in DIR, with the release notes in NOTES_HTML. The app
# reads it from the latest published release (SUFeedURL in scripts/bundle-app.sh).
#
# Sparkle compares sparkle:version with the bundle's CFBundleVersion, which is
# the version.
set -euo pipefail
version="$1"
tag="$2"
dir="$3"
notes_html="$4"
: "${GH_REPO:?}"
script_dir="$(cd "$(dirname "$0")" && pwd)"

dmg="$dir/stemd-$version-macos-arm64.dmg"
[[ -f "$dmg" ]] || { echo "::error::$dmg is missing" >&2; exit 1; }
signature=$("$script_dir/sign.sh" "$dmg")
length=$(wc -c < "$dmg" | tr -d ' ')
url="https://github.com/$GH_REPO/releases/download/$tag/$(basename "$dmg")"
published=$(LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S +0000')

# A CDATA section cannot hold its own terminator.
notes=$(sed 's/]]>/]]]]><![CDATA[>/g' "$notes_html")

cat > "$dir/appcast.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Stemd</title>
    <link>https://github.com/$GH_REPO/releases</link>
    <item>
      <title>Version $version</title>
      <pubDate>$published</pubDate>
      <sparkle:version>$version</sparkle:version>
      <sparkle:shortVersionString>$version</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>14.0</sparkle:minimumSystemVersion>
      <sparkle:fullReleaseNotesLink>https://github.com/$GH_REPO/releases/tag/$tag</sparkle:fullReleaseNotesLink>
      <description><![CDATA[$notes]]></description>
      <enclosure url="$url" length="$length" type="application/octet-stream" sparkle:edSignature="$signature"/>
    </item>
  </channel>
</rss>
XML
cat "$dir/appcast.xml"
