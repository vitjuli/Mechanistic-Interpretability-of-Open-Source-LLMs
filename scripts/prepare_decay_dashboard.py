#!/usr/bin/env python3
"""
Prepare data for dashboard_decay (physics_decay_type interactive dashboard).

Copies from the UI offline run into dashboard_decay/public/data/ and
generates the additional files the dashboard needs:
  - prompts.json      — prompt objects from the JSONL file
  - circuit.json      — circuit summary (from CSD3 circuits JSON if available)
  - community_labels.json — researcher-authored community labels (created if missing)

Usage:
    python scripts/prepare_decay_dashboard.py [--ui_run <run_id>]
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BEHAVIOUR = "physics_decay_type"
DASHBOARD_DATA = PROJECT_ROOT / "dashboard_decay" / "public" / "data"

# Files copied verbatim from UI run
UI_FILES = [
    "interventions.csv",
    "interventions_feature_agg.csv",
    "interventions_layer_agg.csv",
    "interventions_prompt_agg.csv",
    "graph.json",
    "supernodes.json",
    "supernodes_effect.json",
    "supernodes_effect_summary.csv",
    "supernodes_summary.csv",
    "common_prompt_idx.json",
    "run_manifest.json",
    "audit.json",
    "layer_coverage.csv",
]


def find_ui_run(ui_run_arg: str | None) -> Path:
    ui_offline = PROJECT_ROOT / "data" / "ui_offline"
    if ui_run_arg:
        return ui_offline / ui_run_arg
    candidates = sorted(
        d for d in ui_offline.iterdir()
        if d.is_dir() and f"_{BEHAVIOUR}_" in d.name
    )
    if not candidates:
        raise FileNotFoundError(f"No UI run found for {BEHAVIOUR}")
    return candidates[-1]


def copy_ui_files(ui_run_dir: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    missing = []
    for fname in UI_FILES:
        src = ui_run_dir / fname
        if src.exists():
            shutil.copy2(src, dst / fname)
            print(f"  copied: {fname}")
        else:
            missing.append(fname)
    if missing:
        print(f"  WARNING: {len(missing)} files not found: {missing}")


def generate_prompts_json(dst: Path) -> None:
    prompts_jsonl = PROJECT_ROOT / "data" / "prompts" / f"{BEHAVIOUR}_train.jsonl"
    if not prompts_jsonl.exists():
        print(f"  WARNING: {prompts_jsonl} not found — skipping prompts.json")
        return
    prompts = [
        json.loads(l) for l in prompts_jsonl.read_text().splitlines() if l.strip()
    ]
    out = dst / "prompts.json"
    with open(out, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"  generated: prompts.json ({len(prompts)} prompts)")


def generate_circuit_json(dst: Path) -> None:
    # Try to load from CSD3 circuit JSON
    circuit_path = (
        PROJECT_ROOT / "data" / "results" / "causal_edges" / BEHAVIOUR /
        f"circuits_{BEHAVIOUR}_train.json"
    )
    out = dst / "circuit.json"

    if circuit_path.exists():
        shutil.copy2(circuit_path, out)
        print(f"  copied: circuit.json (from {circuit_path.name})")
        return

    # Create a stub with known stats from memory (CSD3 run results)
    stub = {
        "behaviour": BEHAVIOUR,
        "note": "Circuit JSON not available locally (CSD3 only). Stats from SLURM 28420228.",
        "n_features": 11,
        "n_edges": 16,
        "n_paths": 10,
        "disruption_rate": 0.676,
        "s1_sufficiency": {"sign_accuracy": 0.861, "retention": 1.337},
        "s15_sufficiency": {"sign_accuracy": 0.935, "retention": 1.184},
        "top_path": ["input", "L22_F110496", "L23_F83556", "L24_F60777", "L25_F71226", "output_correct"],
        "features": [
            {"id": "L22_F110496", "layer": 22, "feature_idx": 110496},
            {"id": "L23_F83556",  "layer": 23, "feature_idx": 83556},
            {"id": "L23_F71067",  "layer": 23, "feature_idx": 71067},
            {"id": "L24_F60777",  "layer": 24, "feature_idx": 60777},
            {"id": "L24_F52031",  "layer": 24, "feature_idx": 52031},
            {"id": "L24_F18943",  "layer": 24, "feature_idx": 18943},
            {"id": "L24_F88968",  "layer": 24, "feature_idx": 88968},
            {"id": "L24_F249",    "layer": 24, "feature_idx": 249},
            {"id": "L25_F71226",  "layer": 25, "feature_idx": 71226},
            {"id": "L25_F126439", "layer": 25, "feature_idx": 126439},
            {"id": "L25_F110282", "layer": 25, "feature_idx": 110282},
        ],
    }
    with open(out, "w") as f:
        json.dump(stub, f, indent=2)
    print(f"  generated: circuit.json (stub — {stub['n_features']} features from memory)")


def generate_community_labels(dst: Path) -> None:
    out = dst / "community_labels.json"
    if out.exists():
        print(f"  skipped: community_labels.json (already exists — researcher-authored)")
        return
    # Auto-generate sensible initial labels from the Louvain result
    labels = {
        "0": {"label": "Early+mid L10/L15-18", "notes": "Mixed early layers + I/O nodes"},
        "1": {"label": "Isolated pair L22-23",  "notes": "Small cluster; connectivity TBD"},
        "2": {"label": "Late circuit L22-25",   "notes": "Core late-layer circuit features"},
        "3": {"label": "Early L10-14",           "notes": "Early processing layers"},
        "4": {"label": "Bridge L19-22",          "notes": "Narrow bridge; contains L22_F110496"},
        "5": {"label": "Mid L18-22",             "notes": "Mid-network cluster"},
    }
    with open(out, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"  generated: community_labels.json (initial auto-labels — edit as needed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui_run", default=None)
    args = parser.parse_args()

    ui_run_dir = find_ui_run(args.ui_run)
    print(f"Source:      {ui_run_dir.name}")
    print(f"Destination: {DASHBOARD_DATA}")
    print()

    copy_ui_files(ui_run_dir, DASHBOARD_DATA)
    generate_prompts_json(DASHBOARD_DATA)
    generate_circuit_json(DASHBOARD_DATA)
    generate_community_labels(DASHBOARD_DATA)

    print()
    print("Done. Launch with:")
    print("  cd dashboard_decay && npm run dev")


if __name__ == "__main__":
    main()
