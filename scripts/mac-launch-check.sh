#!/usr/bin/env bash
# Launch check of a macOS dmg, on a macOS machine (CI bundle job).
# It does what a user's Mac does with a downloaded dmg: it copies the app out,
# marks it as downloaded (quarantine), asks Gatekeeper, and starts it.
# Usage: scripts/mac-launch-check.sh <path.dmg>
set -uo pipefail
dmg=$1
work=$(mktemp -d)
fail=0
note() { echo "::error::$*"; fail=1; }

mnt="$work/mnt"; mkdir "$mnt"
hdiutil attach -nobrowse -readonly -mountpoint "$mnt" "$dmg" >/dev/null || { note "hdiutil attach failed"; exit 1; }
app_src=$(find "$mnt" -maxdepth 1 -name '*.app' | head -1)
cp -R "$app_src" "$work/"
hdiutil detach "$mnt" >/dev/null
app="$work/$(basename "$app_src")"
exe="$app/Contents/MacOS/$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$app/Contents/Info.plist")"
xattr -w com.apple.quarantine "0081;$(printf %x "$(date +%s)");Safari;" "$app"

echo "== Info.plist"
for k in CFBundleExecutable CFBundleIdentifier LSMinimumSystemVersion NSMicrophoneUsageDescription NSAudioCaptureUsageDescription; do
  v=$(/usr/libexec/PlistBuddy -c "Print :$k" "$app/Contents/Info.plist" 2>/dev/null) || { note "Info.plist has no $k"; continue; }
  echo "$k = $v"
done

echo "== codesign -dv"
codesign -dv --verbose=4 --entitlements - "$app" 2>&1
echo "== codesign --verify --deep --strict"
# A bundle whose signature does not cover its contents is "damaged" for
# Gatekeeper, and macOS gives no way to open it.
codesign --verify --deep --strict --verbose=2 "$app" 2>&1 || note "codesign --verify failed: Gatekeeper reports the app as damaged"
echo "== spctl (Gatekeeper; 'rejected' is expected without a Developer ID)"
spctl -a -vv -t exec "$app" 2>&1 || true

echo "== otool -L"
otool -L "$exe"
# Only system libraries are allowed: the bundle carries no dylib.
bad=$(otool -L "$exe" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/Library/)' || true)
[ -z "$bad" ] || note "links to non-system libraries: $bad"

echo "== launch"
log="$work/launch.log"
"$exe" >"$log" 2>&1 &
pid=$!
sleep 20
if kill -0 "$pid" 2>/dev/null; then
  echo "app runs after 20 s"
  kill "$pid"; wait "$pid" 2>/dev/null
else
  wait "$pid"; note "app stopped within 20 s (exit $?)"
fi
cat "$log"
ls -t ~/Library/Logs/DiagnosticReports 2>/dev/null | head -3 | while read -r r; do echo "== crash report $r"; head -60 ~/Library/Logs/DiagnosticReports/"$r"; done
rm -rf "$work"
exit $fail
