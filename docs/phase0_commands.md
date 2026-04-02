# Phase 0: Baseline Stratification Commands

## Step 1: Compute Baseline Margins

### Local Testing (small subset)
```bash
python scripts/07a_compute_baselines.py \
  --behaviour grammar_agreement \
  --split train \
  --n_prompts 10 \
  --output data/results/baselines/
```

### HPC Production Run
```bash
python scripts/07a_compute_baselines.py \
  --behaviour grammar_agreement \
  --split train \
  --n_prompts 80 \
  --model_name Qwen/Qwen3-4B \
  --output data/results/baselines/
```

**Expected output:** `data/results/baselines/baselines_grammar_agreement_train_n80.csv`

**Expected time:** ~5-10 minutes on CPU

---

## Step 2: Stratify into Low-Margin Subset

```bash
python scripts/07b_stratify_prompts.py \
  --baselines data/results/baselines/baselines_grammar_agreement_train_n80.csv \
  --output data/prompts/low_margin_subset_n40.csv \
  --min_margin 0.0 \
  --max_margin 1.5 \
  --n_per_class 20
```

**Expected output:** `data/prompts/low_margin_subset_n40.csv`

**Expected result:**
- ~40 prompts total (20 singular + 20 plural)
- All with 0 < margin < 1.5
- Balanced classes

---

## Verification

Check the outputs:
```bash
# Check baselines
head -20 data/results/baselines/baselines_grammar_agreement_train_n80.csv

# Count low-margin subset
wc -l data/prompts/low_margin_subset_n40.csv

# Check distribution
python -c "
import pandas as pd
df = pd.read_csv('data/prompts/low_margin_subset_n40.csv')
print(f'Total: {len(df)}')
print(f'Mean margin: {df[\"margin\"].mean():.3f}')
print(df['subject_num'].value_counts())
"
```

---

## SLURM Job Script (HPC)

Create `jobs/phase0_baselines.sh`:

```bash
#!/bin/bash
#SBATCH -J phase0_baselines
#SBATCH -A TURING-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH -o logs/phase0_%j.out
#SBATCH -e logs/phase0_%j.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

source ~/rds/hpc-work/thesis/project/venv/bin/activate

cd ~/rds/hpc-work/thesis/project

# Step 1: Compute baselines
python scripts/07a_compute_baselines.py \
  --behaviour grammar_agreement \
  --split train \
  --n_prompts 80 \
  --output data/results/baselines/

# Step 2: Stratify
python scripts/07b_stratify_prompts.py \
  --baselines data/results/baselines/baselines_grammar_agreement_train_n80.csv \
  --output data/prompts/low_margin_subset_n40.csv \
  --min_margin 0.0 \
  --max_margin 1.5 \
  --n_per_class 20

echo "Phase 0 complete!"
```

Submit:
```bash
sbatch jobs/phase0_baselines.sh
```

---

## Next Steps

After Phase 0 completes:
1. Verify `low_margin_subset_n40.csv` has ~40 balanced prompts
2. Proceed to Phase 0.5: Activation audit
3. Use low-margin subset for all subsequent intervention experiments
