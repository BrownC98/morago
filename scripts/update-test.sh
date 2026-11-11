#!/usr/bin/env bash
# Update test on a CI runner (bundle job). An installed old build (version
# 0.0.1, feature update-test, scripts/update-test.conf.json) finds the new
# release files through a local latest.json, downloads them, checks the
# signature, installs them and starts again, with no user step.
# Usage: scripts/update-test.sh <dist>
#   <dist>: the new signed release files (*-setup.exe or *.app.tar.gz, + .sig).
#   The old build is in src-tauri/target/release/bundle (nsis or macos).
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dist=$(cd "$1" && pwd)
new=$(jq -r .version "$root/src-tauri/tauri.conf.json")
bundle="$root/src-tauri/target/release/bundle"
# No symlink in the path (/var -> /private/var): Tauri refuses such an exe path.
work=$(cd "$(mktemp -d)" && pwd -P)
fail=0
note() { echo "::error::$*"; fail=1; }

case "$(uname -s)" in
  Darwin) os=mac plat=darwin-aarch64 pkg=$(ls "$dist"/*.app.tar.gz) ;;
  *) os=win plat=windows-x86_64 pkg=$(ls "$dist"/*-setup.exe) ;;
esac
name=$(basename "$pkg")
cp "$pkg" "$work/"
jq -n --arg v "$new" --arg p "$plat" --arg url "http://127.0.0.1:8765/$name" --arg sig "$(cat "$pkg.sig")" \
  '{version: $v, pub_date: "2026-01-01T00:00:00Z", platforms: {($p): {url: $url, signature: $sig}}}' > "$work/latest.json"
cat "$work/latest.json"
py=$(command -v python3 || command -v python)
(cd "$work" && exec "$py" -m http.server 8765 --bind 127.0.0.1) >"$work/http.log" 2>&1 &
srv=$!

# Install the old build as a user does.
if [ $os = win ]; then
  "$(ls "$bundle"/nsis/*_0.0.1_*-setup.exe)" /S || note "old installer failed"
  exe="$LOCALAPPDATA/morago/morago.exe"
  version() { powershell -NoProfile -Command "(Get-Item '$(cygpath -w "$exe")').VersionInfo.ProductVersion"; }
  running() { tasklist //FI "IMAGENAME eq morago.exe" | grep -qi morago.exe; }
  stop() { taskkill //F //IM morago.exe >/dev/null 2>&1; }
else
  mkdir "$work/Applications"
  app="$work/Applications/morago.app"
  cp -R "$bundle/macos/morago.app" "$app"
  # A user's copy from a downloaded dmg carries the quarantine mark.
  xattr -w com.apple.quarantine "0081;$(printf %x "$(date +%s)");Safari;" "$app"
  exe="$app/Contents/MacOS/morago"
  version() { /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist"; }
  running() { pgrep -x morago >/dev/null; }
  stop() { pkill -x morago; }
fi
echo "old version: $(version)"
[ "$(version | tr -d '\r')" = 0.0.1 ] || note "the old build is not version 0.0.1"

echo "== update"
MORAGO_UPDATE_TEST="$work/update.log" "$exe" >"$work/app.log" 2>&1 &
ok=0
for _ in $(seq 60); do
  sleep 5
  if [ "$(version 2>/dev/null | tr -d '\r')" = "$new" ] && running; then ok=1; break; fi
done
echo "-- update.log"; cat "$work/update.log" 2>/dev/null
echo "-- http.log"; cat "$work/http.log"
if [ $ok = 1 ]; then
  echo "updated to $new and running"
  sleep 20
  running || note "the updated app stopped within 20 s"
else
  note "no update to $new within 300 s (version now: $(version 2>/dev/null))"
fi
stop; sleep 3

if [ $os = mac ]; then
  echo "== updated bundle"
  xattr -l "$app"
  codesign --verify --deep --strict --verbose=2 "$app" 2>&1 || note "codesign --verify failed on the updated app"
  spctl -a -vv -t exec "$app" 2>&1 || true
  # A start through Launch Services, as from Finder: Gatekeeper blocks a
  # quarantined app with no Developer ID.
  echo "== open"
  open "$app" || note "open failed"
  sleep 15
  if running; then echo "the updated app starts with open"; else note "the updated app does not start with open"; fi
  stop
fi
kill $srv 2>/dev/null
tail -40 "$work/app.log"
rm -rf "$work"
exit $fail
