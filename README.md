# Mechanistic Interpretability of Open Source LLMs
### Reproduction of *On the Biology of a Large Language Model*

**Iuliia Vitiugova**  ·  MPhil in Data Intensive Science  ·  University of Cambridge  ·  July 2026

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Live dashboard:** [vitjuli.github.io/Interpretability-web](https://vitjuli.github.io/Interpretability-web/)

## Description

This repository accompanies the Research Project submission for the MPhil in Data Intensive Science at the University of Cambridge. The coursework instructions are in [Instructions.md](Instructions.md), the project report in [`report/report.pdf`](report/report.pdf), the supplementary material in [`report/supplementary.pdf`](report/supplementary.pdf), and the executive summary in [`report/executive_summary.pdf`](report/executive_summary.pdf).

The project reproduces the circuit-tracing methodology of Ameisen et al. and the analytical programme of [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) on an open model, **Qwen3-4B** with pre-trained per-layer transcoders (`mwhanna/qwen3-4b-transcoders`), and extends it. The central question is whether the direction a linear probe *reads* a concept from is the direction the model *uses* to compute its answer. On a controlled binary task (classifying nuclear decay as alpha or beta) the work shows it is not: the single notion of a "concept direction" separates into three geometrically and causally distinct objects, a **reading** direction (the Fisher discriminant), a **writing** direction (the difference of class means written into the residual stream), and a **use** direction (the gradient of the logit margin). The main result is a double dissociation. The reading direction decodes the class at held-out AUC 0.98–0.99 yet intervening along it leaves the answer unchanged, while the use direction decodes near chance yet reversing it flips the answer.

## Table of Contents
- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Containerisation](#containerisation)
- [Documentation](#documentation)
- [Repository Structure](#repository-structure)
- [Support](#support)
- [License](#license)
- [Project Status](#project-status)
- [Note on the Use of auto-generation tools](#note-on-the-use-of-auto-generation-tools)
- [Authors and Acknowledgment](#authors-and-acknowledgment)

## Data Availability

The small prompt corpora needed to run the pipeline are included under [`data/prompts/`](data/prompts/). Every table and figure in the report is reproduced from the committed files in [`results/`](results/), which do **not** require any external data. Each `results/` subfolder has a `README.md` mapping every file to the report table or figure it supports.

The large intermediate artefacts (≈ 4.2 GB) are excluded from version control (see [`.gitignore`](.gitignore)) and are hosted on Google Drive:

**[Research Project — Submission Data (Google Drive)](https://drive.google.com/drive/folders/1LU5ulYGOckrRhPW_OXK2iMJxA6qXwso8?usp=sharing)**

It contains:

- `field_dump/<concept>/` — per-layer residual and gradient dumps (`res_L*.npy`, `grad_L*.npy`, `meta.npz`) for each concept, the input to the direction-geometry figures. Regenerated with `scripts/119_capture_field_dump.py`.
- `realized_writes/<concept>/` — MLP-vs-attention realized-write dumps, the input to the MLP-share table (`scripts/154`, `scripts/155`).
- `results_reproduction/` — the full circuit-tracing reproduction outputs (attribution graphs, causal edges, transcoder-feature summaries, path validation, candidate-set selection).

The Qwen3-4B base weights and the transcoder dictionary are publicly available from Hugging Face (`Qwen/Qwen3-4B` and `mwhanna/qwen3-4b-transcoders`).

## Installation

### Requirements

- Python 3.10 or higher.
- Conda (recommended) or `venv` for the Python environment.
- A CUDA GPU is required only for the stages that load the model (baseline, feature extraction, field-dump capture, steering). All analysis and plotting stages run on CPU.

### Setup

1. **Clone the repository:**

    ```
    $ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/iv294
    $ cd iv294
    ```

2. **Set up and activate the environment:**

    ```
    $ conda env create -f environment.yml
    $ conda activate qwen-circuits
    ```

    Alternatively, with pip:

    ```
    $ python -m venv .venv && source .venv/bin/activate
    $ pip install -e ".[dev]"
    ```

## Usage

The analysis is a pipeline of numbered, single-purpose scripts in [`scripts/`](scripts/). The GPU stages produce the field dumps and steering sweeps; the CPU stages consume those (or the committed `results/`) to produce every table and figure.

### 1. Circuit-tracing reproduction

```
$ python scripts/02_run_baseline.py                # answer-position fixing, class margins
$ python scripts/04_extract_transcoder_features.py # active transcoder features per prompt
$ python scripts/06_build_attribution_graph.py     # feature-to-feature attribution graph
$ python scripts/07_run_interventions.py           # ablation / activation patching
$ python scripts/08_causal_edges.py                # causal edge identification
$ python scripts/14_path_validation.py             # path-level causal validation
```

### 2. Direction geometry (§3.1–§3.5, Appendices B, C, E)

```
$ python scripts/119_capture_field_dump.py           # GPU: per-layer residual + gradient dumps
$ python scripts/check_decodability_by_layer.py       # -> results/geometry/decodability_by_layer.csv
$ python scripts/axis_geometry_by_layer.py            # -> results/geometry/axis_geometry_by_layer.csv
$ python scripts/110_active_span_triad.py             # -> results/geometry/active_span_triad.csv
$ python scripts/154_dump_realized_writes.py          # GPU: realized writes
$ python scripts/155_realized_write_decomposition.py  # -> results/geometry/realized_*.csv
$ python scripts/118_null_calibration.py              # -> results/geometry/null_calibration/
$ python scripts/133_whitening_theorem.py             # -> results/geometry/whitening/
```

### 3. Direction steering (§3.1.2 causal inertness, §3.3 dissociation)

```
$ python scripts/122_b1_steering_sweep.py             # per-layer x strength x direction margin sweep
$ python scripts/85_steering_all_layers.py            # all-layers steering curve
$ python scripts/steering_summary_table.py            # -> results/steering/steering_summary_all.csv
$ python scripts/steering_decode_check.py             # generations under steering
```

### 4. Feature clustering and selection (§3.6, Appendix D)

```
$ python scripts/22_prepare_clustering_inputs.py      # co-importance (Jaccard) matrix
$ python scripts/23_run_clustering_benchmark.py       # HAC, average linkage
$ python scripts/24_evaluate_clusterings.py           # stability + silhouette sweep (choice of K)
$ python scripts/26_cluster_semantics.py              # physical-feature enrichment (lift)
$ python scripts/build_hac_k12_labels.py              # assemble the HAC K=12 partition
$ python scripts/27_cluster_joint_ablation.py         # causal-impact criterion (joint ablation)
$ python scripts/29_final_cluster_validation.py       # acceptance criteria
```

### 5. Figures

```
$ python scripts/plot_3_1_3_geometry.py     scripts/plot_cos_vs_margin.py \
         scripts/plot_mlp_crossover.py      scripts/plot_norm_ratio.py \
         scripts/plot_feature_span.py       scripts/rotation_decay_by_concept.py
```

## Testing

Unit tests for the reusable library live in [`tests/`](tests/) and run on CPU without the model or the external data:

```
$ pytest -q
```

## Continuous Integration

Every push runs the same checks on both remotes: [`.gitlab-ci.yml`](.gitlab-ci.yml) (Cambridge GitLab) and [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (GitHub). Each has a `lint` stage (`black`, `isort`, `flake8`) and a `test` stage (`pytest`). Style is enforced locally by [`.pre-commit-config.yaml`](.pre-commit-config.yaml):

```
$ pre-commit install
```

## Containerisation

A [`Dockerfile`](Dockerfile) builds a self-contained image for the CPU analysis stages, figures, and tests:

```
$ docker build -t qwen-circuits .
$ docker run --rm qwen-circuits pytest -q
$ docker run --rm -it -v "$PWD":/app qwen-circuits bash
```

## Documentation

API documentation for the `src` library is built with Sphinx:

```
$ pip install -r docs/requirements.txt
$ sphinx-build -b html docs docs/_build
```

## Repository Structure

```
report/                     Compiled report, supplementary and executive summary (PDF)
configs/                    Experiment configuration
data/prompts/               Prompt corpora (small; large data hosted on Google Drive)
src/                        Model helpers, transcoder loader, corpus generators
scripts/                    Numbered pipeline stages and figure scripts
tests/                      CPU unit tests
docs/                       Sphinx API documentation
results/
  geometry/                 §3.1–§3.5 + Appendices B, C, E   (see results/geometry/README.md)
    null_calibration/       covariance-matched null + reliability bound (§3.1.3)
    whitening/              whitening attenuation (Appendix E)
  steering/                 §3.1.2 + §3.3                     (see results/steering/README.md)
  clustering/               §3.6 + Appendix D                 (see results/clustering/README.md)
Dockerfile, .gitlab-ci.yml, .github/, .pre-commit-config.yaml, .flake8
environment.yml, requirements.txt, pyproject.toml, LICENSE
```

## Support

For any questions, feedback, or assistance, please reach out via email at [iv294@cam.ac.uk](mailto:iv294@cam.ac.uk).

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) — see the [LICENSE](LICENSE) file for details.

## Project Status

The project is complete and ready for submission. All essential features are implemented and the codebase is stable.

## Note on the Use of auto-generation tools

Anthropic's Claude (via the Claude Code CLI) was used as a coding and analysis assistant throughout the project: to help write and refactor pipeline scripts, to cross-check numerical results against their source data files, and to assist with editing and consistency-checking of the LaTeX report. All research design, experiments, interpretation, and final wording are the author's own, and every reported number is traceable to a committed result file (see the per-folder `results/*/README.md`).

## Authors and Acknowledgment

This project was written and is maintained by Iuliia Vitiugova (iv294@cam.ac.uk) at the University of Cambridge, MPhil in Data Intensive Science.
