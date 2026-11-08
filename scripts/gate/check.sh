#!/usr/bin/env bash
# Gate check: the Windows + macOS CI run (.github/workflows/ci.yml) must pass.
# No local build (human decision 2026-11-08: local builds starve the shared VM).
# For HEAD it checks the run of HEAD. A local merge commit without a run is
# pushed to a temp branch gate/<sha> so CI builds the merged tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=BrownC98/morago
TIMEOUT=${CI_WAIT_TIMEOUT:-3600}
# Pass rule: no job failed or was cancelled, and "check" passed on Windows and
# macOS. CI skips "check" only when the commit changes no Rust or CI file
# since master (job "changes"). Installers ("bundle") are not part of the gate.
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
  local bad ok skipped
  bad=$(gh run view -R "$REPO" "$id" --json jobs -q '[.jobs[] | select(.conclusion != "success" and .conclusion != "skipped")] | length')
  ok=$(gh run view -R "$REPO" "$id" --json jobs -q '[.jobs[] | select(.conclusion == "success" and (.name | startswith("check (")) and (.name | test("windows|macos")))] | length')
  skipped=$(gh run view -R "$REPO" "$id" --json jobs -q '[.jobs[] | select(.conclusion == "skipped" and (.name | startswith("check")))] | length')
  if [ "$bad" -ne 0 ] || { [ "$ok" -lt 2 ] && [ "$skipped" -eq 0 ]; }; then
    echo "gate: run $id: $bad failed jobs, $ok of 2 'check' jobs passed" >&2
    return 1
  fi
}

head=$(git rev-parse HEAD)
tree=$(git rev-parse "HEAD^{tree}")
if [ -n "$(run_for "$head")" ]; then
  check_sha "$head"
else
  # A parent's green run counts only when HEAD has the same tree (no new code).
  same=""
  for p in $(git rev-list --parents -n 1 HEAD | cut -d' ' -f2-); do
    if [ "$(git rev-parse "$p^{tree}")" = "$tree" ]; then same=$p; break; fi
  done
  if [ -n "$same" ]; then
    echo "gate: HEAD $head has the tree of $same; checking that run"
    check_sha "$same"
  else
    # A real merge: build the merged tree on CI from a temp branch.
    tmp="gate/$head"
    echo "gate: HEAD $head is not on CI; pushing temp branch $tmp"
    git push -q origin "$head:refs/heads/$tmp"
    trap 'git push -q origin --delete "$tmp" || true' EXIT
    check_sha "$head"
  fi
fi
echo "gate: pass"
