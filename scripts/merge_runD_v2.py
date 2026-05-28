import pandas as pd, pathlib, sys

d = pathlib.Path('data/results/interventions/physics_decay_type_probe_v2')
parts = [
    (d / 'runD_v2_p1/intervention_ablation_physics_decay_type_probe_v2.csv', 0),
    (d / 'runD_v2_p2/intervention_ablation_physics_decay_type_probe_v2.csv', 180),
    (d / 'runD_v2_p3/intervention_ablation_physics_decay_type_probe_v2.csv', 360),
]

dfs = []
for path, offset in parts:
    if not path.exists():
        print(f'MISSING: {path}', file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if offset > 0:
        df['prompt_idx'] = df['prompt_idx'] + offset
    print(f'  {path.parent.name}: {len(df)} rows, prompts {df.prompt_idx.min()}-{df.prompt_idx.max()}')
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
out = d / 'runD_v2'
out.mkdir(exist_ok=True)
merged.to_csv(out / 'intervention_ablation_physics_decay_type_probe_v2.csv', index=False)
print(f'Merged: {len(merged)} rows, {merged.prompt_idx.nunique()} prompts, {merged.feature_id.nunique()} features')
