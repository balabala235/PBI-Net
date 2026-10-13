#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
"${PYTHON:-python}" tools/run_experiments.py --variants shift_high shift_low matched_mlp "$@"
