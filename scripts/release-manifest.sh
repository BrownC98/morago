#!/usr/bin/env bash
# Writes the updater manifest (latest.json) for the release files in <dist>.
# The URLs point to the assets of the GitHub release <tag>.
# Usage: scripts/release-manifest.sh <dist> <owner/repo> <tag> > latest.json
set -euo pipefail
dist=$1 repo=$2 tag=$3
base="https://github.com/$repo/releases/download/$tag"
entry() { # <file>
  local f=$1
  [ -f "$f.sig" ] || { echo "no signature for $f" >&2; exit 1; }
  jq -n --arg url "$base/$(basename "$f")" --arg sig "$(cat "$f.sig")" '{url: $url, signature: $sig}'
}
win=$(ls "$dist"/*-setup.exe) mac=$(ls "$dist"/*.app.tar.gz)
jq -n --arg v "${tag#v}" --arg d "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson w "$(entry "$win")" --argjson m "$(entry "$mac")" \
  '{version: $v, pub_date: $d, platforms: {"windows-x86_64": $w, "darwin-aarch64": $m}}'
