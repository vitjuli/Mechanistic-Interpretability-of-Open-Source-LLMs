# Pipeline Execution Guide

## Quick Start

Run the entire pipeline with one command:

```bash
./run_pipeline.sh
```

This will execute all 6 steps sequentially:

1. **Generate Prompts** (`01_generate_prompts.py`)
   - Creates grammar_agreement prompts
   - Output: `data/prompts/grammar_agreement_train.jsonl`

2. **Baseline Evaluation** (`02_run_baseline.py`)
   - Tests model on prompts without interventions
   - Output: `data/results/baseline_grammar_agreement_train.csv`

3. **Extract Transcoder Features** (`04_extract_transcoder_features.py`)
   - Captures MLP inputs and extracts top-k transcoder features
   - Output: `data/results/transcoder_features/layer_*/`

4. **Build Attribution Graph** (`06_build_attribution_graph.py`)
   - Builds feature attribution graph from extracted features
   - Output: `data/results/attribution_graphs/grammar_agreement/`

5. **Run Interventions** (`07_run_interventions.py`)
   - Performs causal interventions on top features
   - Uses **test split** for validation
   - Output: `data/results/interventions/grammar_agreement/`

6. **Generate Figures** (`08_generate_figures.py`)
   - Creates visualizations of results
   - Output: `data/figures/grammar_agreement/`

## On CSD3 (SLURM)

1. **Edit** [`run_pipeline_slurm.sh`](file:///Users/julia/Desktop/courses/thesis/project/run_pipeline_slurm.sh):
   ```bash
   # Line 12: Update venv path
   source /path/to/your/venv/bin/activate
   
   # Line 17: Update cache path
   export HF_HOME=/path/to/your/cache
   
   # Line 23: Update project path
   cd /path/to/project
   ```

2. **Submit job:**
   ```bash
   sbatch run_pipeline_slurm.sh
   ```

3. **Monitor progress:**
   ```bash
   # Watch output
   tail -f logs/pipeline_*.out
   
   # Check job status
   squeue -u $USER
   
   # Check errors
   tail -f logs/pipeline_*.err
   ```

## Resource Requirements

**Local:**
- RAM: ~16GB (for 4B model on CPU)
- Time: ~2-3 hours

**CSD3 (SLURM):**
- CPUs: 8
- RAM: 32GB
- Time: 6 hours (allocated)
- Partition: cclake

## Error Handling

The script uses `set -e`, which means:
- ✅ Pipeline stops immediately if any step fails
- ✅ Easy to identify which step caused the error
- ✅ No wasted compute on downstream steps if upstream fails

**To continue from a specific step:**

Just comment out completed steps in `run_pipeline.sh`:
```bash
# Step 1: Generate prompts (DONE)
# echo "..."
# python scripts/01_generate_prompts.py
# echo "✓ Step 1 complete"

# Step 2: Run baseline (DONE)
# ...

# Step 3: Extract features (STARTING HERE)
echo "=========================================="
echo "STEP 3: Extract Transcoder Features"
...
```

## Manual Execution (Step-by-Step)

If you prefer to run steps individually:

```bash
# Step 1
python scripts/01_generate_prompts.py

# Step 2
python scripts/02_run_baseline.py --behaviour grammar_agreement --split train

# Step 3
python scripts/04_extract_transcoder_features.py \
    --behaviour grammar_agreement \
    --split train

# Step 4
python scripts/06_build_attribution_graph.py \
    --behaviour grammar_agreement \
    --split train \
    --n_prompts 20

# Step 5 — STRICT (graph-driven only, default): skips layers with no graph features;
# exits with error if zero interventions are produced. Use this for publishable results.
python scripts/07_run_interventions.py \
    --behaviour grammar_agreement \
    --split train \
    --graph_n_prompts 80

# Step 5 — CONTROL (fallback enabled): layers with no graph features use first-K features.
# Output CSVs contain feature_source='control' rows — filter these out before publishing.
python scripts/07_run_interventions.py \
    --behaviour grammar_agreement \
    --split train \
    --graph_n_prompts 80 \
    --control_fallback

# Step 6
python scripts/08_generate_figures.py \
    --behaviour grammar_agreement
```

## Strict vs Control Mode (script 07)

Script 07 runs in **strict mode** by default:

- If a layer has no features in the attribution graph → the layer is **skipped** with a WARNING.
- If **all** layers are skipped (zero interventions produced) → the script **exits with code 1**.
- Output `feature_source` column is always `"graph"` in strict mode.

To run a **control experiment** (old behaviour, first-K features as baseline):

```bash
python scripts/07_run_interventions.py --behaviour grammar_agreement --control_fallback
```

When `--control_fallback` is active:
- A large WARNING is printed at startup.
- Layers with no graph features use the first-K features instead of skipping.
- Output CSVs include a `feature_source` column: `"graph"` or `"control"`.
- Filter with `df[df.feature_source == "graph"]` to get only graph-driven rows.
- The `layer_has_graph_features` column records whether the layer had any graph features.
- `control_fallback: true` is written into every `_summary.json` file for audit.

**Never mix strict and control results without the `feature_source` filter.**


## Output Structure

After successful completion:

```
data/
├── prompts/
│   └── grammar_agreement_train.jsonl
├── results/
│   ├── baseline_grammar_agreement_train.csv
│   ├── transcoder_features/
│   │   ├── layer_15/
│   │   ├── layer_16/
│   │   └── ...
│   ├── attribution_graphs/
│   │   └── grammar_agreement/
│   └── interventions/
│       └── grammar_agreement/
└── figures/
    └── grammar_agreement/
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
- Activate your virtual environment first
- Install requirements: `pip install -r requirements.txt`

**"FileNotFoundError: prompts not found"**
- Make sure Step 1 completed successfully
- Check `data/prompts/` directory exists

**"CUDA out of memory"**
- Pipeline is configured for CPU (`device: "cpu"` in config)
- If using GPU, reduce batch sizes in `experiment_config.yaml`

**Empty attribution graph**
- Check that script 04 completed successfully
- Verify `data/results/transcoder_features/` has files
- See [`empty_graph_diagnosis.md`](file:///Users/julia/.gemini/antigravity/brain/2ba81278-fdcb-4e90-ad1a-ebfb78257cd9/empty_graph_diagnosis.md) for details
