import pandas as pd, pathlib

d = pathlib.Path('data/results/interventions/physics_decay_type_probe_v2')
p1 = pd.read_csv(d / 'runC_p1/intervention_ablation_physics_decay_type_probe_v2.csv')
p2 = pd.read_csv(d / 'runC_p2/intervention_ablation_physics_decay_type_probe_v2.csv')
p2['prompt_idx'] += 270
merged = pd.concat([p1, p2], ignore_index=True)
(d / 'runC').mkdir(exist_ok=True)
merged.to_csv(d / 'runC/intervention_ablation_physics_decay_type_probe_v2.csv', index=False)
print(f'Merged: {len(merged)} rows, {merged.prompt_idx.nunique()} prompts')
