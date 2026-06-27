#!/bin/bash
# Hardness diagnostic — wraps 142 with all 5 ρ-dumps.

set -uo pipefail

cd "$(dirname "$0")/.."

DUMPS_DIR=data/analysis/runD_v2/forced_crossover

python scripts/142_forced_crossover_hardness.py \
    --dumps \
        "rho1.00=${DUMPS_DIR}/dump_rho1.00/field_dump" \
        "rho0.75=${DUMPS_DIR}/dump_rho0.75/field_dump" \
        "rho0.50=${DUMPS_DIR}/dump_rho0.50/field_dump" \
        "rho0.25=${DUMPS_DIR}/dump_rho0.25/field_dump" \
        "rho0.00=${DUMPS_DIR}/dump_rho0.00/field_dump"
