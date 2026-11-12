#!/usr/bin/env bash
# Install and use check of a released macOS dmg, on a macOS CI runner
# (ci.yml, manual job `mac-install`). It does what a user does: download the
# dmg from the GitHub release, copy the app to /Applications, open it, and
# press the controls of the window through the accessibility API (ax.swift).
# Usage: scripts/mac-install/check.sh <tag> <out-dir>
#   <out-dir> gets results.tsv (check, pass|fail|info, note), screenshots, logs.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tag=$1
out=$(mkdir -p "$2" && cd "$2" && pwd)
repo=${GITHUB_REPOSITORY:-BrownC98/morago}
app=/Applications/morago.app
fail=0
res() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$out/results.tsv"; [ "$2" != fail ] || fail=1; }
shot() { screencapture -x "$out/$1.png" 2>>"$out/screencapture.log" || true; }
pid() { pgrep -x morago | head -1; }
crashes() { ls ~/Library/Logs/DiagnosticReports 2>/dev/null | grep -i morago || true; }

echo "== runner"
{
  sw_vers; uname -m; sysctl -n machdep.cpu.brand_string hw.memsize
  echo "-- gatekeeper"; spctl --status 2>&1; csrutil status 2>&1
  echo "-- displays"; system_profiler SPDisplaysDataType 2>&1
  echo "-- audio"; system_profiler SPAudioDataType 2>&1
} | tee "$out/runner.txt"

echo "== ax tool"
swiftc -O -o "$out/ax" "$here/ax.swift" 2>&1 | tail -5
ax() { "$out/ax" "$@"; }
ax trusted
# Lines of the window that match a pattern.
has() { ax dump "$(pid)" 2>/dev/null | grep -E -- "$1"; }
# Waits until the window has (or, with `gone`, no longer has) a matching line.
wait_for() { # <pattern> <seconds> [gone]
  local i
  for ((i = 0; i < $2; i += 2)); do
    if has "$1" >/dev/null; then [ "${3:-}" = gone ] || return 0; else [ "${3:-}" != gone ] || return 0; fi
    sleep 2
  done
  return 1
}
press() { ax press "$(pid)" "$@"; }

echo "== 1. install"
dmg="$out/morago.dmg"
url=$(curl -fsSL "https://api.github.com/repos/$repo/releases/tags/$tag" | jq -r '.assets[] | select(.name | endswith(".dmg")) | .browser_download_url')
curl -fsSL --retry 5 -o "$dmg" "$url" && res download pass "$url ($(stat -f %z "$dmg") bytes, sha256 $(shasum -a 256 "$dmg" | cut -c1-16)…)" || res download fail "$url"
# A browser marks its downloads; the mark goes to the files of the mounted image.
xattr -w com.apple.quarantine "0083;$(printf %x "$(date +%s)");Safari;$(uuidgen)" "$dmg"
mnt=$(mktemp -d)
hdiutil attach -nobrowse -readonly -mountpoint "$mnt" "$dmg" >"$out/hdiutil.txt" 2>&1 || res mount fail "$(tail -1 "$out/hdiutil.txt")"
ls -la "$mnt" | tee -a "$out/hdiutil.txt"
rm -rf "$app"
cp -R "$mnt/morago.app" /Applications/ && res copy pass "morago.app → /Applications" || res copy fail "cp failed"
hdiutil detach "$mnt" >/dev/null
rm "$dmg"
{
  echo "-- xattr"; xattr -l "$app"
  echo "-- codesign -dv"; codesign -dv --verbose=4 "$app" 2>&1
  echo "-- codesign --verify"; codesign --verify --deep --strict --verbose=2 "$app" 2>&1; echo "exit $?"
  echo "-- spctl --assess"; spctl --assess -vv -t exec "$app" 2>&1; echo "exit $?"
  echo "-- version"; /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist"
} | tee "$out/signature.txt"
[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")" = "${tag#v}" ] && res version pass "${tag#v}" || res version fail "Info.plist version is not ${tag#v}"
codesign --verify --deep --strict "$app" 2>/dev/null && res codesign pass "$(grep -E '^Signature|^TeamIdentifier' "$out/signature.txt" | tr '\n' ' ')" || res codesign fail "codesign --verify failed: macOS calls the app damaged"
res spctl info "$(spctl --assess -vv -t exec "$app" 2>&1 | tr '\n' ' ')"
xattr -p com.apple.quarantine "$app" >/dev/null 2>&1 && res quarantine info "the copied app carries the quarantine mark" || {
  res quarantine info "cp did not carry the mark from the dmg; set it on the app"
  xattr -w com.apple.quarantine "0083;$(printf %x "$(date +%s)");Safari;$(uuidgen)" "$app"
}

echo "== 1b. first open with the quarantine mark (Gatekeeper)"
open "$app" 2>&1 | tee "$out/open-quarantined.txt"
sleep 15
shot 1-open-quarantined
for p in CoreServicesUIAgent UserNotificationCenter; do
  q=$(pgrep -x $p | head -1) && { echo "-- $p"; ax dump "$q"; }
done >"$out/gatekeeper-dialog.txt" 2>&1
cat "$out/gatekeeper-dialog.txt"
if [ -n "$(pid)" ]; then
  res gatekeeper info "the quarantined app started with no dialog (Gatekeeper: $(spctl --status 2>&1))"
  pkill -x morago; sleep 3
else
  res gatekeeper info "macOS did not start the quarantined app: $(grep -E 'AXStaticText' "$out/gatekeeper-dialog.txt" | sed 's/.*|//' | tr '\n' ' ' | cut -c1-300)"
  pkill -x CoreServicesUIAgent
fi
# What "Open Anyway" in System Settings does: it takes the mark off.
xattr -dr com.apple.quarantine "$app"

echo "== 2. start"
open --env RUST_LOG=info --stdout "$out/app.log" --stderr "$out/app.log" "$app" || res open fail "open failed"
sleep 10
shot 2-start-10s
sleep 50
shot 2-start-60s
if [ -n "$(pid)" ] && [ -z "$(crashes)" ]; then res start pass "runs after 60 s, no crash report"; else res start fail "pid='$(pid)' crash reports: $(crashes)"; fi
ax dump "$(pid)" >"$out/ax-start.txt" 2>&1
wc -l "$out/ax-start.txt"; cat "$out/ax-start.txt"

echo "== probes"
{
  echo "-- TCC user db"; sqlite3 "$HOME/Library/Application Support/com.apple.TCC/TCC.db" 'select service, client, auth_value from access' 2>&1
  echo "-- TCC system db"; sudo sqlite3 "/Library/Application Support/com.apple.TCC/TCC.db" 'select service, client, auth_value from access' 2>&1
  echo "-- app files"; ls -la ~/Library/Application\ Support/com.morago.app ~/Library/Logs/com.morago.app ~/Documents/morago 2>&1
} | tee "$out/probes.txt"

pkill -x morago
cat "$out/app.log" 2>/dev/null | tail -100
crashes
rm -f "$out/ax"
exit $fail
