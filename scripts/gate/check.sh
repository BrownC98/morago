#!/usr/bin/env bash
# Local checks before a merge. CI (.github/workflows/ci.yml) also builds Windows and macOS.
set -euo pipefail
cd "$(dirname "$0")/../.."
pnpm install --frozen-lockfile
pnpm build
cd src-tauri
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
