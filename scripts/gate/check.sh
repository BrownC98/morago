#!/usr/bin/env bash
# Gate check: the Windows + macOS CI run (.github/workflows/ci.yml) must pass.
# No local build (human decision 2026-11-08: local builds starve the shared VM).
# For HEAD it checks the run of HEAD; if HEAD is a local merge commit with no
# run, it checks the run of each parent instead. Push each commit first.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=BrownC98/morago
TIMEOUT=${CI_WAIT_TIMEOUT:-3600}

run_for() { # prints the newest ci.yml run id for a commit, or nothing
  gh run list -R "$REPO" --workflow ci.yml --commit "$1" -L 1 \
    --json databaseId -q '.[0].databaseId // empty'
}

check_sha() {
  local sha=$1 id=""
  for _ in $(seq 1 30); do # a fresh push needs a moment to start its run
    id=$(run_for "$sha")
    [ -n "$id" ] && break
    sleep 10
  done
  if [ -z "$id" ]; then
    echo "gate: no CI run for $sha. Push the commit first." >&2
    return 1
  fi
  echo "gate: CI run https://github.com/$REPO/actions/runs/$id for $sha"
  timeout "$TIMEOUT" gh run watch -R "$REPO" "$id" --exit-status >/dev/null || {
    echo "gate: CI run $id failed or timed out" >&2
    return 1
  }
  local ok
  ok=$(gh run view -R "$REPO" "$id" --json jobs \
    -q '[.jobs[] | select(.conclusion == "success") | .name] | map(select(test("windows|macos"))) | length')
  if [ "$ok" -lt 2 ]; then
    echo "gate: run $id lacks a successful Windows and macOS job" >&2
    return 1
  fi
}

head=$(git rev-parse HEAD)
if [ -n "$(run_for "$head")" ]; then
  check_sha "$head"
else
  parents=$(git rev-list --parents -n 1 HEAD | cut -d' ' -f2-)
  [ "$(wc -w <<<"$parents")" -ge 2 ] || { check_sha "$head"; exit; }
  echo "gate: HEAD $head is a local merge; checking its parents"
  for p in $parents; do check_sha "$p"; done
fi
echo "gate: pass"
