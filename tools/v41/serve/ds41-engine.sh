#!/usr/bin/env bash
# Optional foreground shim: STRATA_V41_ENGINE=/path/to/ds41_generate ds41-engine.sh --serve ...
# No workstation-specific GPU locks, systemd scopes, memory caps or DSpark sidecars.
set -eu
if [ -z "${STRATA_V41_ENGINE:-}" ]; then
    echo "Set STRATA_V41_ENGINE to your built ds41_generate executable." >&2
    exit 2
fi
exec "$STRATA_V41_ENGINE" "$@"
