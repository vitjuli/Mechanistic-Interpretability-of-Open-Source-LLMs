# Mechanistic Interpretability of Open Source LLMs — Reproduction of On the Biology of a Large Language Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This project is associated with the submission of the coursework for the Research Project as part of the MPhil in Data Intensive Science at the University of Cambridge. The coursework instructions can be found under [Instructions.md](Instructions.md). The associated project report can be found under [Project Report](report/report.pdf). The associated executive summary can be found under [Executive Summary](report/executive_summary.pdf).

The project reproduces the circuit-tracing methodology of Ameisen et al. and the analytical programme of [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) on an open model, **Qwen3-4B** with pre-trained per-layer transcoders (`mwhanna/qwen3-4b-transcoders`), and extends it. The central question is whether the direction a linear probe *reads* a concept from is the direction the model *uses* to compute its answer. On a controlled binary task (classifying nuclear decay as alpha or beta) the work shows it is not: the single notion of a "concept direction" separates into three geometrically and causally distinct objects, a **reading** direction (the Fisher discriminant), a **writing** direction (the difference of class means written into the residual stream), and a **use** direction (the gradient of the logit margin). The main result is a double dissociation, the reading direction decodes the class at held-out AUC 0.98–0.99 yet intervening along it leaves the answer unchanged, while the use direction decodes near chance yet reversing it flips the answer.

## Table of Contents
- [Data Availability](#data-availability)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Support](#support)
- [License](#license)
- [Project Status](#project-status)
- [Note on the Use of auto-generation tools](#note-on-the-use-of-auto-generation-tools)
- [Authors and Acknowledgment](#authors-and-acknowledgment)

## Data Availability

The small prompt corpora needed to run the pipeline are included in the repository under [`data/prompts/`](data/prompts/). The large intermediate artefacts are **not** committed (see [`.gitignore`](.gitignore)) and are available here:

**[Research Project — Submission Data](TODO: add Google Drive link)**

It includes the following:

- `field_dump/` — per-layer residual and gradient dumps (`res_L00..35.npy`, `grad_L00..35.npy`, `meta.npz`) for each concept, the input to all direction-geometry analysis. Regenerated from scratch with `scripts/119_capture_field_dump.py`.
- `models/` — the Qwen3-4B base weights and the per-layer transcoder dictionary. These are also publicly available from Hugging Face (`Qwen/Qwen3-4B` and `mwhanna/qwen3-4b-transcoders`).

The final result files that reproduce every table and figure in the report are committed under [`results/`](results/) and do **not** require the external data.

## Installation

### Requirements

- Python 3.10 or higher.
- Conda (for managing the Python environment).
- A CUDA GPU is required only for the stages that load the model (baseline, feature extraction, field-dump capture, steering). All analysis and plotting stages run on CPU.

### Setup

1. **Clone the repository:**

    ```
    $ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/iv294
    ```

2. **Navigate to the project directory:**

    ```
    $ cd iv294
    ```

3. **Set up and activate the environment:**

    ```
    $ conda env create -f environment.yml
    $ conda activate qwen-circuits
    ```

    Alternatively, with pip:

    ```
    $ python -m venv .venv && source .venv/bin/activate
    $ pip install -r requirements.txt
    ```

## Usage

The analysis is a pipeline of numbered, single-purpose scripts in [`scripts/`](scripts/). The GPU stages produce the field dumps and steering sweeps; the CPU stages consume those (or the committed `results/`) to produce every table and figure. Each `results/` subfolder has a `README.md` mapping every file to the report table or figure it supports.

### 1. Circuit-tracing reproduction (report Chapter on reproduction)

```
$ python scripts/02_run_baseline.py               # answer-position fixing, class margins
$ python scripts/04_extract_transcoder_features.py # active transcoder features per prompt
$ python scripts/06_build_attribution_graph.py     # feature-to-feature attribution graph
$ python scripts/07_run_interventions.py           # ablation / activation patching
$ python scripts/08_causal_edges.py                # causal edge identification
$ python scripts/14_path_validation.py             # path-level causal validation
```

### 2. Direction geometry (§3.1–§3.5, Appendices B, C, E)

```
$ python scripts/119_capture_field_dump.py          # GPU: per-layer residual + gradient dumps
$ python scripts/check_decodability_by_layer.py      # -> results/geometry/decodability_by_layer.csv
$ python scripts/axis_geometry_by_layer.py           # -> results/geometry/axis_geometry_by_layer.csv
$ python scripts/110_active_span_triad.py            # -> results/geometry/active_span_triad.csv
$ python scripts/154_dump_realized_writes.py         # GPU: realized writes
$ python scripts/155_realized_write_decomposition.py # -> results/geometry/realized_*.csv
$ python scripts/133_whitening_theorem.py            # -> results/geometry/whitening/
```

### 3. Direction steering (§3.1.2 causal inertness, §3.3 dissociation)

```
$ python scripts/122_b1_steering_sweep.py            # per-layer x strength x direction margin sweep
$ python scripts/85_steering_all_layers.py           # all-layers steering curve
$ python scripts/steering_summary_table.py           # -> results/steering/steering_summary_all.csv
$ python scripts/steering_decode_check.py            # generations under steering
```

### 4. Feature clustering (§3.6, Appendix D)

```
$ python scripts/22_prepare_clustering_inputs.py     # co-importance (Jaccard) matrix
$ python scripts/23_run_clustering_benchmark.py      # HAC, average linkage
$ python scripts/24_evaluate_clusterings.py          # stability + silhouette sweep
$ python scripts/26_cluster_semantics.py             # physical-feature enrichment
$ python scripts/build_hac_k12_labels.py             # assemble the HAC k=12 partition
$ python scripts/27_cluster_joint_ablation.py        # causal-impact criterion (joint ablation)
$ python scripts/29_final_cluster_validation.py      # acceptance criteria
```

## Repository Structure

```
report/                     Compiled report and executive summary (PDF)
configs/                    Experiment configuration
data/prompts/               Prompt corpora (small; large data hosted externally)
src/                        Model helpers, transcoder loader, corpus generators
scripts/                    Numbered pipeline stages
results/
  geometry/                 §3.1–§3.5 + Appendices B, C, E  (see results/geometry/README.md)
  steering/                 §3.1.2 + §3.3                    (see results/steering/README.md)
  clustering/               §3.6 + Appendix D                (see results/clustering/README.md)
environment.yml, requirements.txt, pyproject.toml, LICENSE
```

## Support

For any questions, feedback, or assistance, please reach out via email at [iv294@cam.ac.uk](mailto:iv294@cam.ac.uk).

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) — see the [LICENSE](LICENSE) file for details.

## Project Status

The project is in a state ready for submission. All essential features have been implemented and the codebase is stable.

## Note on the Use of auto-generation tools

<!-- TODO (author to verify and finalise): the declaration below must accurately reflect the tools you used. -->

Anthropic's Claude (via the Claude Code CLI) was used as a coding and analysis assistant throughout the project: to help write and refactor pipeline scripts, to cross-check numerical results against their source data files, and to assist with editing and consistency-checking of the LaTeX report. All research design, experiments, interpretation, and final wording are the author's own, and every reported number is traceable to a committed result file (see the per-folder `results/*/README.md`).

## Authors and Acknowledgment

This project was written and is maintained by Iuliia Vitiugova at the University of Cambridge (MPhil in Data Intensive Science).
