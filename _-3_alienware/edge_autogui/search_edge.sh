#!/usr/bin/env bash
set -euo pipefail

# --- Resolve paths and share the scheduler's run lock.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
state_dir="$HOME/.local/state/edge-autogui"
mkdir -p -- "$state_dir"
exec 9>"$state_dir/search.lock"
flock -n 9 || { echo "Another Edge search run is active." >&2; exit 1; }

# --- Start now; arguments supplied by the caller override these defaults.
exec python3 "$script_dir/edge_autogui.py" \
    --repeat 20 --wait-min 5 --wait-max 15 --start-delay 0 "$@"
