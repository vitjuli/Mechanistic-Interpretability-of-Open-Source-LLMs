# Mechanistic Interpretability of Qwen3 Models
## A Reproducibility Study of "On the Biology of a Large Language Model"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Project Overview

This project investigates whether circuit motifs identified in Anthropic's [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) generalize to smaller, open-weight models. We use **pre-trained transcoders** from the [circuit-tracer](https://github.com/safety-research/circuit-tracer) project to analyze **Qwen3 models** (0.6B to 14B parameters).

**Central question:** Are interpretable circuits fundamental algorithmic primitives, or scale-dependent emergent phenomena?

### Scientific Approach

1. **Behaviour selection:** Focus on 4 behaviours where circuits are theoretically likely:
   - **Grammatical number agreement** (subject-verb)
   - **Factual recall** (country-capital knowledge)
   - **Sentiment continuation** (positive/negative text completion)
   - **Arithmetic** (two-digit addition)

2. **Pre-trained transcoders:** Use per-layer transcoders (PLTs) from HuggingFace to decompose MLP computations into interpretable features

3. **Attribution graphs:** Build causal dependency graphs from inputs -> transcoder features -> outputs

4. **Causal validation:** Ablation and patching experiments to verify circuit structure

5. **Multi-scale analysis:** Compare circuits across model sizes (0.6B, 1.7B, 4B, 8B, 14B)

**Negative results are scientifically valid.** If circuits don't exist at smaller scales, that's a publishable finding about capability emergence.

---

## Available Pre-trained Transcoders

| Model Size | HuggingFace Repository | Layers |
|------------|------------------------|--------|
| 0.6B | [mwhanna/qwen3-0.6b-transcoders-lowl0](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0) | 28 |
| 1.7B | [mwhanna/qwen3-1.7b-transcoders-lowl0](https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0) | 28 |
| 4B | [mwhanna/qwen3-4b-transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders) | 36 |
| 8B | [mwhanna/qwen3-8b-transcoders](https://huggingface.co/mwhanna/qwen3-8b-transcoders) | 36 |
| 14B | [mwhanna/qwen3-14b-transcoders-lowl0](https://huggingface.co/mwhanna/qwen3-14b-transcoders-lowl0) | 40 |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd project

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Test Setup

```bash
# Verify installation (CPU-only)
python scripts/test_setup.py --skip-model

# With GPU
python scripts/test_setup.py
```

### Run Pipeline

```bash
# 1. Generate synthetic prompts for all 4 behaviours
python scripts/01_generate_prompts.py

# 2. Measure baseline performance (requires GPU)
python scripts/02_run_baseline.py --all --split train

# 3. Extract transcoder features
python scripts/04_extract_transcoder_features.py --all --model_size 4b

# 4. Build attribution graphs
python scripts/06_build_attribution_graph.py --all --model_size 4b

# 5. Run intervention experiments
python scripts/07_run_interventions.py --all --model_size 4b

# 6. Generate figures
python scripts/08_generate_figures.py --all
```

---

## Repository Structure

```
project/
├── configs/
│   ├── experiment_config.yaml    # Core experiment settings
│   └── transcoder_config.yaml    # Transcoder-specific settings
├── data/
│   ├── prompts/                  # Synthetic prompt sets (4 behaviours)
│   ├── activations/              # Captured hidden states (gitignored)
│   └── results/                  # Outputs, metrics, visualizations
├── figures/                      # Publication-quality plots
├── scripts/
│   ├── 01_generate_prompts.py           # Generate prompts
│   ├── 02_run_baseline.py               # Baseline evaluation
│   ├── 03_capture_activations.py        # Extract activations (legacy)
│   ├── 04_extract_transcoder_features.py # Extract transcoder features
│   ├── 05_interpret_features.py         # Feature interpretation
│   ├── 06_build_attribution_graph.py    # Build causal graphs
│   ├── 07_run_interventions.py          # Ablation/patching
│   ├── 08_generate_figures.py           # Publication figures
│   └── test_setup.py                    # Verify installation
├── src/
│   ├── model_utils.py            # Qwen model wrapper
│   ├── transcoder/               # Transcoder loading and utilities
│   │   ├── __init__.py
│   │   ├── activation_functions.py  # JumpReLU, TopK
│   │   ├── single_layer_transcoder.py
│   │   └── transcoder_loader.py     # HuggingFace loading
│   ├── attribution.py            # Attribution graph utilities
│   ├── interventions.py          # Intervention methods
│   └── visualization.py          # Plotting utilities
├── archive/                      # Legacy SAE training code
│   ├── legacy_sae/
│   └── legacy_scripts/
├── slurm_scripts/                # CSD3 HPC job scripts
├── notebooks/                    # Interactive analysis
└── docs/                         # Additional documentation
```

---

## Experimental Pipeline

### Phase 1: Prompt Generation & Baseline
**Scripts:** `01_generate_prompts.py`, `02_run_baseline.py`

Generate synthetic prompts and establish baseline model performance. Only proceed with behaviours achieving ≥80% accuracy.

### Phase 2: Feature Extraction
**Script:** `04_extract_transcoder_features.py`

Load pre-trained transcoders from HuggingFace and extract feature activations for analysis layers.

```bash
python scripts/04_extract_transcoder_features.py \
    --behaviour grammar_agreement \
    --model_size 4b \
    --layers 10 11 12 13 14 15 16 17 18 19 20
```

### Phase 3: Attribution Graphs
**Script:** `06_build_attribution_graph.py`

Build causal dependency graphs showing how features contribute to outputs using gradient-based attribution.

### Phase 4: Circuit Validation
**Script:** `07_run_interventions.py`

Validate identified circuits through:
- **Feature ablation:** Zero or suppress specific features
- **Activation patching:** Swap features between prompt pairs
- **Feature importance:** Correlate feature activations with logit differences

### Phase 5: Visualization
**Script:** `08_generate_figures.py`

Generate publication-quality figures for thesis.

---

## Running on Cambridge CSD3

### Setup

```bash
# SSH to CSD3
ssh <user>@login.hpc.cam.ac.uk

# Load modules
module load rhel8/default-amp
module load python/3.10.5-gcc-11.2.0
module load cuda/12.1

# Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Submit Jobs

```bash
# Full pipeline
sbatch slurm_scripts/pipeline_full.slurm

# Individual steps
sbatch slurm_scripts/01_baseline.slurm
sbatch slurm_scripts/04_attribution_graph.slurm
sbatch slurm_scripts/05_interventions.slurm
```

---

## Behaviours Analyzed

| Behaviour | Description | Success Threshold |
|-----------|-------------|-------------------|
| Grammar Agreement | Subject-verb number agreement (singular/plural) | 80% accuracy |
| Factual Recall | Country-capital factual knowledge | 80% accuracy |
| Sentiment Continuation | Sentiment-consistent text completion | 75% accuracy |
| Arithmetic | Two-digit addition | 80% accuracy |

---

## Comparison to Anthropic's Work

| Dimension | Anthropic (2025) | This Work |
|-----------|------------------|-----------|
| Model | Claude Sonnet 3 (~100B) | Qwen3 (0.6B-14B) |
| Model access | Proprietary | Open weights |
| Feature extraction | Cross-layer transcoders (CLTs) | Per-layer transcoders (PLTs) |
| Training | Trained on proprietary data | Pre-trained (circuit-tracer) |
| Behaviours | 20+ | 4 (focused) |
| Compute | Production-scale | Single A100 GPU |
| Multi-scale | Single model | 5 model sizes |

---

## Key Differences from Original Methodology

1. **Pre-trained transcoders instead of trained SAEs:** We use transcoders from the circuit-tracer project rather than training our own. This provides:
   - Reproducibility (same features across researchers)
   - Computational efficiency (no training required)
   - Larger feature dictionaries

2. **Per-layer transcoders (PLTs) instead of cross-layer transcoders (CLTs):**
   - PLTs decompose single MLP layers
   - CLTs have multi-layer decoders enabling direct cross-layer attribution
   - We approximate cross-layer effects via virtual weight matrices

3. **Multiple model scales:** We analyze circuits across 5 model sizes to study scale-dependent phenomena.

---

## Reproducibility Standards

- **Seeds:** All random operations use fixed seeds (see `configs/experiment_config.yaml`)
  - Prompt generation: 42
  - Interventions: 456
  - PyTorch: 789
- **Environment:** Pinned dependencies in `requirements.txt`
- **Data:** Synthetic prompts committed; features downloadable
- **Code:** All figures generated from scripts (no manual editing)

---

## Key References

1. Lindsey, J., Gurnee, W., et al. (2025). [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). Anthropic Transformer Circuits Thread.

2. Anthropic (2025). [Attribution Graphs: Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Technical methodology.

3. Safety Research (2025). [circuit-tracer](https://github.com/safety-research/circuit-tracer). GitHub repository.

4. Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv.

---

## Citation

```bibtex
@misc{qwen3circuits2025,
  title={Mechanistic Interpretability of Qwen3 Models: A Multi-Scale Reproducibility Study},
  author={[Your Name]},
  year={2025},
  institution={University of Cambridge, DAMTP},
  howpublished={\url{https://github.com/[repo-url]}}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Supervisor

Dr Miles Cranmer
Assistant Professor, DAMTP & Institute of Astronomy
University of Cambridge
mc2473@cam.ac.uk
