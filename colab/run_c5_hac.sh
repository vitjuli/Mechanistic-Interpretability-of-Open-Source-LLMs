#!/usr/bin/env bash
# c5 causal on the HAC k=12 partition (Table 9). One short call from Colab:
#   !bash colab/run_c5_hac.sh
# Does: build hac_k12 column -> joint ablation on HAC clusters 1..12 -> analysis.
set -e
cd /content/project 2>/dev/null || cd "$(dirname "$0")/.."
echo "== cwd: $(pwd) =="

python scripts/build_hac_k12_labels.py

python -u scripts/27_cluster_joint_ablation.py \
    --behaviour physics_decay_type_probe \
    --split train \
    --clusters all \
    --device cuda \
    --clustering_dir data/results/clustering_hac \
    --cluster_col hac_k12

echo "== ablation done; running analysis (27b) =="
python scripts/27b_analyse_joint_ablation.py \
    --behaviour physics_decay_type_probe \
    --split train || echo "check 27b args (ablation output already saved)"
