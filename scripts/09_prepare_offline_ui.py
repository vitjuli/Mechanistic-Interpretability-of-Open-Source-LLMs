#!/usr/bin/env python3
"""
Prepare offline UI-ready dataset from attribution graphs + intervention results.

Produces a versioned directory of parquet/csv/json artifacts suitable for
Neuronpedia-style dashboards, interactive filtering, supernodes, drill-down.

No GPU, no model, no torch required.

Usage:
    python scripts/09_prepare_offline_ui.py \
        --behaviour grammar_agreement \
        --split train \
        --graph_n_prompts 80

    python scripts/09_prepare_offline_ui.py \
        --behaviour grammar_agreement \
        --split train \
        --graph_n_prompts 80 \
        --run_id my_custom_run \
        --community_method louvain \
        --effect_clusters 30
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui_offline.prepare import prepare_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare offline UI-ready dataset from intervention results.",
    )
    parser.add_argument(
        "--behaviour", type=str, default="grammar_agreement",
        help="Behaviour name (default: grammar_agreement)",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Data split (default: train)",
    )
    parser.add_argument(
        "--graph_n_prompts", type=int, default=80,
        help="n_prompts suffix for attribution graph file (default: 80)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/ui_offline",
        help="Output base directory (default: data/ui_offline)",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Run identifier. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--community_method", type=str, default="louvain",
        choices=["louvain", "components"],
        help="Community detection method for graph supernodes (default: louvain)",
    )
    parser.add_argument(
        "--effect_clusters", type=int, default=None,
        help="Number of effect-similarity clusters (default: auto = min(50, sqrt(n)))",
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment_config.yaml",
        help="Path to experiment config for results path",
    )
    args = parser.parse_args()

    # Load config for results path
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        results_dir = Path(config["paths"]["results"])
    else:
        results_dir = Path("data/results")
        logger.warning(f"Config not found at {config_path}, using default: {results_dir}")

    # Auto-generate run_id
    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_id = f"{ts}_{args.behaviour}_{args.split}_n{args.graph_n_prompts}"

    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("OFFLINE UI DATA PREPARATION")
    print("=" * 70)
    print(f"  Behaviour:       {args.behaviour}")
    print(f"  Split:           {args.split}")
    print(f"  Graph n_prompts: {args.graph_n_prompts}")
    print(f"  Run ID:          {args.run_id}")
    print(f"  Output:          {out_dir / args.run_id}")
    print(f"  Community:       {args.community_method}")
    print(f"  Effect clusters: {args.effect_clusters or 'auto'}")
    print("=" * 70)

    run_dir = prepare_all(
        results_dir=results_dir,
        behaviour=args.behaviour,
        split=args.split,
        graph_n_prompts=args.graph_n_prompts,
        out_dir=out_dir,
        run_id=args.run_id,
        community_method=args.community_method,
        effect_clusters=args.effect_clusters,
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Output: {run_dir}")
    print(f"\n  Next: python scripts/09_sanity_check_ui_data.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
