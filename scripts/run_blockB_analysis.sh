#!/bin/bash
# Block B analysis — wraps 141 with all 5 ρ-dumps + scaffold reference.

set -uo pipefail

cd "$(dirname "$0")/.."

DUMPS_DIR=data/analysis/runD_v2/forced_crossover

python scripts/141_forced_crossover_blockB.py \
    --dumps \
        "rho1.00=${DUMPS_DIR}/dump_rho1.00/field_dump" \
        "rho0.75=${DUMPS_DIR}/dump_rho0.75/field_dump" \
        "rho0.50=${DUMPS_DIR}/dump_rho0.50/field_dump" \
        "rho0.25=${DUMPS_DIR}/dump_rho0.25/field_dump" \
        "rho0.00=${DUMPS_DIR}/dump_rho0.00/field_dump" \
    --scaffold_dump data/analysis/runD_v2/B1_alpha_beta/field_dump \
    --cue_rho 1.0 \
    --concept_rho 0.5 \
    --mid_lo 18 --mid_hi 28 \
    --readout_lo 33 --readout_hi 35 \
    --out "${DUMPS_DIR}/blockB_trajectory.csv"
