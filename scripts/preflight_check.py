"""Static pre-flight: parse each `python scripts/X.py --flags` call in a job file and verify
every --flag is actually declared by X.py, and every required=True arg of X.py is supplied.
Catches the arg-mismatch crashes before submitting. No imports of the heavy scripts."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def declared_flags(script_path):
    """Return (all_flags:set, required_flags:set) from a script's argparse calls."""
    txt = script_path.read_text()
    allf, req = set(), set()
    for m in re.finditer(r'add_argument\(\s*"(--[a-zA-Z0-9_]+)"(.*?)\)', txt, re.S):
        flag, rest = m.group(1), m.group(2)
        allf.add(flag)
        if "required=True" in rest:
            req.add(flag)
    return allf, req

def invocations(job_text):
    """Yield (script_rel, [flags]) for each `python ... scripts/X.py ...` (handles \\ line-continuation)."""
    # join backslash-continued lines
    joined = re.sub(r"\\\s*\n", " ", job_text)
    for line in joined.splitlines():
        m = re.search(r"python[0-9]*\s+(?:-u\s+)?(scripts/[\w/]+\.py)(.*)", line)
        if not m:
            continue
        script, tail = m.group(1), m.group(2)
        flags = re.findall(r"(--[a-zA-Z0-9_]+)", tail)
        yield script, flags

def check(job_file):
    jf = ROOT / job_file
    if not jf.exists():
        print(f"  !! job file missing: {job_file}"); return False
    ok = True
    for script, used in invocations(jf.read_text()):
        sp = ROOT / script
        if not sp.exists():
            print(f"  !! {script} NOT FOUND"); ok = False; continue
        allf, req = declared_flags(sp)
        bad = [f for f in used if f not in allf]
        missing_req = [f for f in req if f not in used]
        tag = "OK " if not bad and not missing_req else "FAIL"
        print(f"  [{tag}] {script}")
        if bad:
            print(f"         unknown flags (WILL CRASH): {bad}"); ok = False
        if missing_req:
            print(f"         missing required args (WILL CRASH): {missing_req}"); ok = False
    return ok

INPUTS = [
    ("data/prompts/particle_pairs", "6 sliced pair files (part_pairs jobs)", 6),
    ("data/prompts/physics_decay_type_probe_v2_train.jsonl", "27c prompts", None),
    ("data/results/clustering/cluster_membership_ch5.csv", "27c cluster membership", None),
    ("data/results/clustering/cluster_labels.csv", "27c pool227", None),
    ("data/prompts/physics_internal_candidate_selection_v2_train.jsonl", "reserve corpus (clusters)", None),
]

def check_inputs():
    ok = True
    for path, desc, n_expect in INPUTS:
        p = ROOT / path
        if not p.exists():
            print(f"  !! MISSING: {path}  ({desc})"); ok = False; continue
        if n_expect is not None:
            n = len(list(p.glob("particles_*.jsonl")))
            tag = "OK " if n == n_expect else "FAIL"
            if n != n_expect: ok = False
            print(f"  [{tag}] {path}  ({n}/{n_expect} files — {desc})")
        else:
            print(f"  [OK ] {path}  ({desc})")
    return ok

if __name__ == "__main__":
    jobs = sys.argv[1:] or [
        "jobs/run_particle_pairs_all.sbatch",
        "jobs/run_27c_jobs.sbatch",
        "jobs/run_particle_clusters_gpu.sbatch",
        "scripts/run_particle_clusters_local.sh",
    ]
    allok = True
    for j in jobs:
        print(f"\n=== {j} ===")
        allok &= check(j)
    print("\n=== INPUT FILES ===")
    allok &= check_inputs()
    print("\n" + ("ALL PASS — args + inputs OK" if allok else "FAILURES above — fix before submit"))
    sys.exit(0 if allok else 1)
