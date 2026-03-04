# Run Index

Frozen snapshots of completed pipeline runs.
Each entry links to a self-contained folder with prompts, results, and README.

---

## physics_scalar_vector_operator — train, n=80 (2026-02-24)

| Field | Value |
|---|---|
| **Run ID** | `20260224-115852_physics_scalar_vector_operator_train_n80` |
| **Date** | 2026-02-24 |
| **Behaviour** | `physics_scalar_vector_operator` |
| **Split** | train |
| **n_prompts (graph)** | 80 |
| **Git commit** | `2f68795` (fix,gpu) |
| **Snapshot folder** | [`runs/physics_scalar_vector_operator_train_n80_20260224/`](physics_scalar_vector_operator_train_n80_20260224/) |

**Summary:** Physics scalar/vector operator classification (Type 3: Abstraction). Qwen3-4B-Instruct-2507 achieves 83.75% baseline accuracy (mean logprob_diff=4.44). Attribution graph: 23 feature nodes across layers 13–25, 69 links. All three intervention types run (ablation, patching, steering).

---

## multilingual_antonym — train, n=80 (PENDING — submit jobs/multilingual_02_09.sbatch)

| Field | Value |
|---|---|
| **Run ID** | TBD after job completes |
| **Date** | 2026-03-04 (Stage A committed), GPU run pending |
| **Behaviour** | `multilingual_antonym` |
| **Split** | train |
| **n_prompts (graph)** | 80 |
| **Git commit** | `5d592e9` (Stage A), see also Stage B commit |
| **Snapshot folder** | TBD — will be `runs/multilingual_antonym_train_n80_YYYYMMDD/` |
| **SLURM job** | `jobs/multilingual_02_09.sbatch` |

**Prompt pool (104 total, 80 train / 24 test):**
- EN antonym: 9 concepts × 4 templates (concept_idx 0-8)
- FR antonym: 8 concepts × 4 templates (concept_idx 0,2-8; hot/cold excluded: froid=2 tokens)
- EN synonym: 9 concepts × 4 templates (tiny, warm, quick, fresh, bare, tall, lengthy, neat, simple)

**Three intervention axes (Anthropic multilingual circuits reproduction):**
- C1 operation swap: `--experiment patching --patch_mode C1` (antonym features → synonym context)
- C2 operand swap:   `--experiment patching --patch_mode C2` (hot→small, EN only)
- C3 language swap:  `--experiment patching --patch_mode C3` (EN→FR routing, 8 concepts)
- Ablation: `--top_k 20` (fixes coverage from 59.7% → ~100% by covering all graph features)

**Success criteria:**
- Baseline: EN antonym ≥ 90%, EN synonym ≥ 60%, FR antonym ≥ 80%
- Attribution graph: features across layers 10-25
- C1: ≥ 50% pairs with effect_size < 0 (synonym margin decreases when antonym features patched)
- C2: ≥ 40% pairs with effect_size < 0 (small→large margin decreases)
- C3: ≥ 40% pairs with sign_flipped=True or effect_size < 0 (FR margin decreases)
- Ablation coverage: ≥ 95% of graph features tested (verify with a_analyze_interventions_coverage.py)
