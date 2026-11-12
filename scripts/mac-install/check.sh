#!/usr/bin/env bash
# Install and use check of a released macOS dmg, on a macOS CI runner
# (ci.yml, manual job `mac-install`). It does what a user does: download the
# dmg from the GitHub release, copy the app to /Applications, open it, and
# press the controls of the window through the accessibility API (ax.swift).
# The signature checks of a dmg before its release are in mac-launch-check.sh.
# Usage: scripts/mac-install/check.sh <out-dir> [<tag>]
#   <out-dir> gets results.tsv (check, pass|fail|info, note), screenshots, logs.
#   <tag>: default, the latest release (needs GH_TOKEN).
# The runner (3 CPUs, a virtual GPU) plays sound late while the translation
# model works: the meeting checks need one utterance with a translation, not
# every clip (runs 43184650099, 43187081403).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
repo=${GITHUB_REPOSITORY:-BrownC98/morago}
tag=${2:-$(gh release view -R "$repo" --json tagName --jq .tagName)}
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
# `open` and a dump of a busy app can wait without end: give each a time limit.
tmo() { perl -e 'alarm shift; exec @ARGV' "$@"; }
ax() { tmo 120 "$out/ax" "$@"; }
ax trusted
# Lines of the window that match a pattern.
has() { ax dump "$(pid)" 2>/dev/null | grep -E -- "$1"; }
# Waits until the window has (or, with `gone`, no longer has) a matching line.
# `gone` needs the web page in the dump: an empty dump (no app, a time limit)
# is not "gone".
wait_for() { # <pattern> <seconds> [gone]
  local i d
  for ((i = 0; i < $2; i += 2)); do
    d=$(ax dump "$(pid)" 2>/dev/null)
    if grep -qE -- "$1" <<<"$d"; then
      [ "${3:-}" = gone ] || return 0
    elif [ "${3:-}" = gone ] && grep -q AXWebArea <<<"$d"; then
      return 0
    fi
    sleep 2
  done
  return 1
}
press() { ax press "$(pid)" "$@"; }

echo "== sound device"
# The runner has no sound device. BlackHole is a loopback device: what plays
# on its output comes in on its input, so `afplay` is the meeting's speech.
brew install --cask blackhole-2ch >"$out/blackhole.txt" 2>&1 || tail -20 "$out/blackhole.txt"
sudo killall coreaudiod 2>/dev/null || sudo launchctl kickstart -kp system/com.apple.audio.coreaudiod
sleep 5
brew install switchaudio-osx >>"$out/blackhole.txt" 2>&1
SwitchAudioSource -t input -s "BlackHole 2ch"; SwitchAudioSource -t output -s "BlackHole 2ch"
system_profiler SPAudioDataType 2>&1 | tee "$out/audio-devices.txt"
[ "$(SwitchAudioSource -c -t input)$(SwitchAudioSource -c -t output)" = "BlackHole 2chBlackHole 2ch" ] && res sound-device info "BlackHole 2ch is the default input and output device" || res sound-device info "no loopback device: a meeting gets no sound"

echo "== 1. install"
dmg="$out/morago.dmg"
# The link of the release page, as a browser gets it.
url="https://github.com/$repo/releases/download/$tag/morago_${tag#v}_aarch64.dmg"
curl -fsSL --retry 5 -o "$dmg" "$url" && res download pass "$url ($(stat -f %z "$dmg") bytes, sha256 $(shasum -a 256 "$dmg" | cut -c1-16)…)" || res download fail "$url"
# A browser marks its downloads; the mark goes to the files of the mounted image.
xattr -w com.apple.quarantine "0083;$(printf %x "$(date +%s)");Safari;$(uuidgen)" "$dmg"
mnt=$(mktemp -d)
hdiutil attach -nobrowse -readonly -mountpoint "$mnt" "$dmg" >"$out/hdiutil.txt" 2>&1 || res mount fail "$(tail -1 "$out/hdiutil.txt")"
ls -la "$mnt" | tee -a "$out/hdiutil.txt"
rm -rf "$app"
cp -R "$mnt/morago.app" /Applications/ && res copy pass "morago.app → /Applications" || res copy fail "cp failed"
hdiutil detach "$mnt" >/dev/null
[ -d "$app" ] || { echo "::error::no app to check"; exit 1; }
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
# `open` did not return in run 43176261368: it waits while macOS holds the app.
tmo 600 open "$app" >"$out/open-quarantined.txt" 2>&1 &
sleep 20
shot 1-open-quarantined
ps -axo pid,stat,etime,comm | grep -E 'morago|CoreServicesUIAgent|UserNotificationCenter|SecurityAgent|open$' | tee "$out/gatekeeper-ps.txt"
for p in CoreServicesUIAgent UserNotificationCenter SecurityAgent; do
  for q in $(pgrep -x $p); do echo "-- $p $q"; ax dump "$q"; done
done >"$out/gatekeeper-dialog.txt" 2>&1
cat "$out/gatekeeper-dialog.txt"
if [ -n "$(pid)" ] && tmo 30 "$out/ax" dump "$(pid)" | grep -q AXWindow; then
  res gatekeeper info "the quarantined app started with no dialog (Gatekeeper: $(spctl --status 2>&1))"
  pkill -x morago; sleep 3
else
  res gatekeeper info "macOS did not start the quarantined app: $(grep -E 'AXStaticText' "$out/gatekeeper-dialog.txt" | sed 's/.*|//' | tr '\n' ' ' | cut -c1-300)"
  pkill -x morago; pkill -x open
  ax press "$(pgrep -x CoreServicesUIAgent | head -1)" AXButton "=Done"; sleep 3
fi
# What "Open Anyway" in System Settings does: it takes the mark off.
xattr -dr com.apple.quarantine "$app"

echo "== 2. start"
launch() { tmo 60 open --env RUST_LOG=info --stdout "$out/app.log" --stderr "$out/app.log" "$app" || res open fail "open failed or did not return in 60 s"; }
launch
sleep 10
ax fit "$(pid)" 1024 700
shot 2-start-10s
sleep 50
shot 2-start-60s
if [ -n "$(pid)" ] && [ -z "$(crashes)" ]; then res start pass "runs after 60 s, no crash report"; else res start fail "pid='$(pid)' crash reports: $(crashes)"; fi

echo "== 3. use"
# Permission dialogs of macOS (microphone, system audio, Documents folder):
# keep a picture and the text of each, then press Allow as a user does.
(
  n=0
  while sleep 3; do
    for p in UserNotificationCenter CoreServicesUIAgent; do
      for q in $(pgrep -x $p); do
        d=$(tmo 30 "$out/ax" dump "$q" 2>/dev/null)
        b=$(grep -oE 'AXButton\|[^|]*\|on\|(Allow|OK|허용|확인)$' <<<"$d" | head -1) || continue
        n=$((n + 1)); shot "prompt-$n"
        { echo "-- prompt $n ($p)"; echo "$d"; } >>"$out/prompts.txt"
        tmo 30 "$out/ax" press "$q" AXButton "=${b##*|}" >>"$out/prompts.txt" 2>&1
      done
    done
  done
) &
prompts=$!
step() { # <check> <seconds> <pattern that must show> [gone]: result line + tree + picture
  if wait_for "$3" "$2" "${4:-}"; then res "$1" pass "${5:-window shows: $3 ${4:-}}"; else res "$1" fail "window: no '$3' ${4:-} in $2 s"; fi
  ax dump "$(pid)" >"$out/ax-$1.txt" 2>&1; shot "3-$1"
}
step ui-tree 60 'AXWebArea'

# Models: the two notices on the meeting tab lead to the settings tab.
press AXButton "설정에서 내려받기"
step settings-tab 20 'SenseVoice Small 내려받기'
press AXButton "SenseVoice Small 내려받기"
press AXButton "Gemma 4 E2B 내려받기"
sleep 5; shot 3-downloading
t0=$(date +%s)
step model-download 1800 '(SenseVoice Small|Gemma 4 E2B) 내려받기' gone
res model-download-time info "SenseVoice Small and Gemma 4 E2B in $(( $(date +%s) - t0 )) s"
# The updater asks the real release manifest. An older tag has an update.
press AXButton "업데이트 확인"
step update-check 60 '최신 버전입니다|새 버전 [0-9.]+이 있습니다'
models=~/Library/Application\ Support/com.morago.app/models
find "$models" -type f -exec shasum -a 256 {} + 2>&1 | tee "$out/models.txt"
bad=0; n=0
while read -r sha file; do
  n=$((n + 1)); grep -q "\"$sha\"" "$here/../../src-tauri/src/models.rs" || { bad=1; echo "no pin for $file"; }
done <"$out/models.txt"
[ $n -ge 3 ] && [ $bad = 0 ] && res model-sha pass "$n files match the SHA-256 pins of models.rs" || res model-sha fail "$n files, mismatch=$bad"

# Meeting with the recording on.
press AXRadioButton "=회의" || press '*' "=회의"
step meeting-tab 20 '회의 시작'
press AXCheckBox "WAV 파일로 저장"
sleep 2
settings=~/Library/Application\ Support/com.morago.app/settings.json
cat "$settings" | tee "$out/settings.json"; echo
[ "$(jq -r .record_audio "$settings")" = true ] && res settings-save pass "the 녹음 switch wrote record_audio=true to settings.json" || res settings-save fail "settings.json has no record_audio=true"
press AXButton "회의 시작"
# The virtual GPU of the runner can time out while the model loads (run
# 43190180879); the app then says so and stays ready: start once more.
if ! wait_for '중지하고 저장' 240 && has '회의를 시작하지 못했습니다' >"$out/meeting-start-error.txt"; then
  shot 3-meeting-start-error
  res meeting-start-retry info "$(sed 's/.*|//' "$out/meeting-start-error.txt" | cut -c1-200)"
  press AXButton "회의 시작"
fi
step meeting-start 240 '중지하고 저장'
mkdir -p "$out/wavs"
hf=https://huggingface.co/csukuangfj
curl -fsSL --retry 5 -o "$out/wavs/en.wav" $hf/sherpa-onnx-whisper-base/resolve/main/test_wavs/0.wav
curl -fsSL --retry 5 -o "$out/wavs/ja.wav" $hf/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ja.wav
curl -fsSL --retry 5 -o "$out/wavs/ko.wav" $hf/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/test_wavs/ko.wav
for w in en ja ko; do afplay "$out/wavs/$w.wav" 2>&1; sleep 8; done
rm -rf "$out/wavs"
shot 3-meeting-after-speech
# Until a translation is in the session file (the model loads at the first one).
for ((i = 0; i < 300; i += 5)); do
  grep -qs '"state":"done"' ~/Documents/morago/sessions/*/utterances.jsonl && break
  sleep 5
done
sleep 20
ax dump "$(pid)" >"$out/ax-meeting.txt" 2>&1; shot 3-meeting
sed -n '/발화 목록/,$p' "$out/ax-meeting.txt" | grep AXStaticText | sed 's/.*|//' | tee "$out/meeting-text.txt"
press AXButton "중지하고 저장"
step meeting-stop 180 '회의 시작'

sessions=~/Documents/morago/sessions
find "$sessions" -type f -exec ls -l {} + 2>&1 | tee "$out/session-files.txt"
dir=$(ls -d "$sessions"/*/ 2>/dev/null | head -1)
cp "$dir"/meta.json "$dir"/utterances.jsonl "$dir"/audio.wav "$out/" 2>/dev/null
u=$(wc -l <"$dir/utterances.jsonl" 2>/dev/null | tr -d ' ')
[ "${u:-0}" -ge 1 ] && res session-save pass "meta.json and utterances.jsonl with $u lines" || res session-save fail "no utterance in $dir"
jq -r 'select(.translation.state == "done") | .translation.text' "$dir/utterances.jsonl" 2>/dev/null | grep -q '[가-힣]' && res translation pass "utterances have a Korean translation" || res translation fail "no Korean translation in utterances.jsonl"
afinfo "$dir/audio.wav" 2>&1 | tee "$out/audio-wav.txt"
grep -q '1 ch,  16000 Hz' "$out/audio-wav.txt" && [ "$(stat -f %z "$dir/audio.wav")" -gt 320000 ] && res recording pass "audio.wav: mono 16 kHz, $(stat -f %z "$dir/audio.wav") bytes" || res recording fail "audio.wav is missing, short or not mono 16 kHz"

# Records tab: the saved meeting, its player, the folder button.
press AXRadioButton "=기록" || press '*' "=기록"
step records-tab 20 '폴더 열기'
grep -q '회의 녹음' "$out/ax-records-tab.txt" && res records-player pass "the records tab shows the recording player" || res records-player fail "no player (회의 녹음) on the records tab"
# Export: the save dialog of macOS, with its default folder and name.
for b in "=Markdown" "녹음(WAV)"; do
  press AXButton "$b"
  wait_for 'AXButton.*\|Save$' 20 || res export fail "no save dialog for $b"
  shot "3-export-$(tr -dc 'A-Za-z' <<<"$b")"
  press AXButton "=Save"
  wait_for '내보냈습니다' 30 || res export fail "no '내보냈습니다' after Save for $b"
  press AXButton "알림 닫기"; sleep 1
done
find ~/Documents ~/Desktop ~/Downloads -maxdepth 2 -type f \( -name '*.md' -o -name '*.wav' \) -not -path '*/sessions/*' -exec ls -l {} + 2>/dev/null | tee "$out/export-files.txt"
md=$(grep -o '/.*\.md$' "$out/export-files.txt" | head -1); wav=$(grep -o '/.*\.wav$' "$out/export-files.txt" | head -1)
[ -n "$md" ] && cp "$md" "$out/export.md"
[ -s "$md" ] && grep -q '[가-힣]' "$md" && res export-markdown pass "$md" || res export-markdown fail "no Markdown file"
[ -n "$wav" ] && cmp -s "$wav" "$dir/audio.wav" && res export-audio pass "$wav equals audio.wav" || res export-audio fail "no WAV file equal to audio.wav"
press AXButton "폴더 열기"
sleep 5; shot 3-folder-open
f=$(osascript -e 'tell application "Finder" to get POSIX path of (target of front window as alias)' 2>&1)
case "$f" in "$sessions"*) res folder-open pass "Finder shows $f" ;; *) res folder-open fail "Finder front window: $f" ;; esac

# Quit as a user does, start again: the settings and the record stay.
press AXMenuItem "Quit morago"
sleep 8
[ -z "$(pid)" ] && res quit pass "the app ends on Quit" || { res quit fail "the app still runs 8 s after Quit"; pkill -x morago; sleep 3; }
launch
step restart 60 '회의 시작'
grep -qE 'AXCheckBox.*WAV 파일로 저장.* 1$' "$out/ax-restart.txt" && res settings-kept pass "the 녹음 switch is still on after a restart" || res settings-kept fail "the 녹음 switch is off after a restart"
kill $prompts 2>/dev/null
[ -f "$out/prompts.txt" ] && res prompts info "$(grep -c '^-- prompt' "$out/prompts.txt") macOS permission dialogs; see prompts.txt" || res prompts info "macOS showed no permission dialog"

echo "== end"
{
  echo "-- TCC user db"; sqlite3 "$HOME/Library/Application Support/com.apple.TCC/TCC.db" "select service, client, auth_value from access where client like '%morago%'" 2>&1
} | tee "$out/probes.txt"
pkill -x morago; sleep 2
[ -z "$(crashes)" ] && res crash-reports pass "no crash report of morago" || { res crash-reports fail "$(crashes)"; cp ~/Library/Logs/DiagnosticReports/*orago* "$out/" 2>/dev/null; }
# Runs 43190180879 and 43197148234: the virtual GPU of the runner timed out
# under the translation model; after it the screen can stay white.
grep -q kIOGPUCommandBufferCallbackErrorTimeout "$out/app.log" && res runner-gpu info "the runner's virtual GPU timed out: run the job again"
grep -vE '^ggml_metal_library_compile_all' "$out/app.log" | tail -150
cat "$out/results.tsv"
rm -f "$out/ax"
exit $fail
