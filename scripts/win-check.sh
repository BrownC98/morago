#!/usr/bin/env bash
# Rust checks on the Windows host of this PC, run from WSL.
#   scripts/win-check.sh               the CI check job: clippy --all-targets -D warnings, cargo test
#   scripts/win-check.sh check         cargo check
#   scripts/win-check.sh test engine   cargo test engine (cargo arguments as given)
# The work tree (also uncommitted files) goes as a commit to one fixed git
# repository on C:. No build over \\wsl$: it is slow. The target folder is on
# C: and all worktrees share it. One run at a time (flock). The exit code is
# that of the last cargo command.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
win_home=$(wslpath "$(cmd.exe /d /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")
root="$win_home/morago-win-check"
wroot=$(wslpath -w "$root")
mkdir -p "$root/sherpa-onnx" "$HOME/.cache"

exec 9>"$HOME/.cache/morago-win-check.lock"
flock -n 9 || { echo "win-check: waiting for another run" >&2; flock 9; }

# 1. Snapshot: all files not in .gitignore go into a temp index. No ref or index of the worktree changes.
idx=$(mktemp); trap 'rm -f "$idx"' EXIT
cp "$(git -C "$repo" rev-parse --path-format=absolute --git-path index)" "$idx"
GIT_INDEX_FILE=$idx git -C "$repo" add -A
sha=$(git -C "$repo" commit-tree -p HEAD -m win-check "$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree)")

# 2. Push to the C: repository, check out with Windows git (no line-end change). Ignored files (target) stay.
if [ ! -d "$root/repo/.git" ]; then
  git.exe init -q "$wroot\\repo"
  git.exe -C "$wroot\\repo" config core.autocrlf false
fi
git -C "$repo" push -q --force "$root/repo" "$sha:refs/heads/check"
git.exe -C "$wroot\\repo" checkout -q -f --detach check
git.exe -C "$wroot\\repo" clean -q -ffd

# 3. sherpa-onnx prebuilt libs: the build script downloads with no token and got
#    504 (2026-11-08). Get the archive once through the API with the gh token and
#    give SHERPA_ONNX_ARCHIVE_DIR (same method as .github/actions/sherpa-libs).
v=$(grep -A1 '^name = "sherpa-onnx-sys"' "$repo/src-tauri/Cargo.lock" | sed -n 's/^version = "\(.*\)"/\1/p')
f="sherpa-onnx-v$v-win-x64-static-MT-Release-lib.tar.bz2"
if [ ! -s "$root/sherpa-onnx/$f" ]; then
  id=$(gh api "repos/k2-fsa/sherpa-onnx/releases/tags/v$v" --jq ".assets[] | select(.name == \"$f\") | .id")
  curl -fsSL --retry 8 --retry-all-errors --retry-max-time 600 \
    -H "Authorization: Bearer $(gh auth token)" -H "Accept: application/octet-stream" \
    -o "$root/sherpa-onnx/$f.part" "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/assets/$id"
  mv "$root/sherpa-onnx/$f.part" "$root/sherpa-onnx/$f"
fi

# 4. Run cargo on Windows. The PATH from WSL can be older than the installs, so add the tool folders.
if [ $# -eq 0 ]; then
  steps=("cargo clippy --all-targets -- -D warnings" "cargo test")
else
  steps=("cargo $*")
fi
{
  echo '@echo off'
  echo 'set "PATH=%USERPROFILE%\.cargo\bin;%ProgramFiles%\LLVM\bin;%PATH%"'
  # winget installs CMake as a portable zip; its command alias did not appear.
  for d in "$win_home"/AppData/Local/Microsoft/WinGet/Packages/Kitware.CMake_*/cmake-*/bin; do
    if [ -d "$d" ]; then echo "set \"PATH=$(wslpath -w "$d");%PATH%\""; fi
  done
  echo 'set "LIBCLANG_PATH=%ProgramFiles%\LLVM\bin"'
  # As in CI: no debug info. The test binaries then link in less time.
  echo 'set CARGO_PROFILE_DEV_DEBUG=0'
  echo 'set CARGO_PROFILE_TEST_DEBUG=0'
  echo "set \"SHERPA_ONNX_ARCHIVE_DIR=$wroot\\sherpa-onnx\""
  # CUDA build of llama.cpp (Cargo.toml feature "cuda"): CUDA toolkit (CUDA_PATH),
  # MSVC environment and Ninja. Its own short target folder: the CPU build of
  # other worktrees does not rebuild it, and nvcc gets short paths.
  if grep -q 'features = \["cuda"\]' "$repo/src-tauri/Cargo.toml"; then
    vs=$("/mnt/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" \
      -latest -products '*' -property installationPath 2>/dev/null | tr -d '\r')
    echo 'if not exist "%CUDA_PATH%\bin\nvcc.exe" (echo win-check: CUDA toolkit not found ^(CUDA_PATH^) & exit /b 1)'
    echo "call \"$vs\\VC\\Auxiliary\\Build\\vcvars64.bat\" >nul"
    echo "set \"PATH=$vs\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\Ninja;%PATH%\""
    echo 'set CMAKE_GENERATOR=Ninja'
    echo 'set "CARGO_TARGET_DIR=%SystemDrive%\mct"'
  fi
  echo "cd /d \"$wroot\\repo\\src-tauri\""
  for s in "${steps[@]}"; do
    echo "echo ^> $s"
    echo "$s"
    echo 'if errorlevel 1 exit /b %errorlevel%'
  done
} | sed 's/$/\r/' > "$root/run.cmd"
start=$SECONDS
cd "$root"
rc=0; cmd.exe /d /c "$wroot\\run.cmd" || rc=$?
echo "win-check: exit $rc, $((SECONDS - start)) s (${sha:0:12})" >&2
exit $rc
