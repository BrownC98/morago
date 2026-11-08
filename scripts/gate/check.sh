#!/usr/bin/env bash
# Local checks before a merge. CI (.github/workflows/ci.yml) also builds Windows and macOS.
set -euo pipefail
cd "$(dirname "$0")/../.."
# The loom gate shell has no fnm/rustup setup: add the usual tool folders.
for d in "$HOME/.local/share/fnm/aliases/default/bin" "$HOME/.cargo/bin"; do
  [ -d "$d" ] && PATH="$d:$PATH"
done
export PATH
pnpm install --frozen-lockfile
pnpm build
cd src-tauri
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
