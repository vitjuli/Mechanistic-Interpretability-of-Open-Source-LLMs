# physics_conservation v1 — ARCHIVED 2026-03-26

Archived because: prompts were unstable (True/False bias), baseline unreliable.
Replaced by: scalar/vector behavior for demo.

## What's here
- prompts/: All JSONL prompt files (pilot, pilot_v3, train, test)
- jobs/: SLURM job scripts for running the pipeline
- diag_format_test.py: Diagnostic script for format testing

## History
- v1: True/False statement format → systematic True bias
- v2: "True or False? ... Answer:" format → passed pilot (83.3%)
- Full pipeline run: SLURM 26214149 → FAIL (56%) — JSONL not regenerated after format fix
- v3 pilot (Yes/No questions): implemented but NOT run; abandoned for demo

## To restore
1. Re-run: python scripts/01_generate_prompts.py --behaviour physics_conservation
2. Re-run full pipeline: jobs/physics_conservation_02_09.sbatch
