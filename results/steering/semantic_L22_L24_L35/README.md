# Semantic steering bundle (L22 / L24 / L35)

Per-prompt semantic-steering logs at three depths (L22, L24, L35), produced by
`colab/run_semantic_steering.sh`. For each prompt the log records the baseline
correct-vs-incorrect logit margin and, for the three directions, the margin
change under steering across a range of strengths, together with a short
generation.

## Contents

- `<pair>_L{22,24,35}.txt` — per-prompt logs. Each prompt line gives
  `correct=`, `incorrect=` and `base_margin` (the correct-minus-incorrect logit
  margin), then the margin delta `dm` for `delta` (writing `Δμ`), `usage` (use
  `u`) and `w_res` (reading `w_LDA`) at strengths `c = 1, 2, 4, 8, 16, 32`,
  followed by a 30-token generation.
- `semantic_steering_all.csv` — the same records in aggregate
  (`tag, layer, c, prompt_idx, correct, incorrect, push, margin, base_margin,
  dmargin, flipped, top1, top5, gen`).

Pairs: `alpha_beta`, `grammar`, `electron_vs_proton`, `neutron_vs_proton`. For
every retained file the `incorrect` class matches the pair named in the file, so
`base_margin` is the correct-minus-paired-class margin.

## Relation to the thesis

Provenance for the §3.3 "Dissociation under intervention" generation figure,
which shows the neutron/proton generations at L24 (`neutron_vs_proton_L24.txt`):
steering along the use direction forces the class token and degenerates into
repetition, while the writing and reading directions keep the continuation
coherent.

These logs are a separate, later run from the per-pair margin-flip sweeps in the
parent directory (`steering_pair_*.csv`), which are the source of the §3.1.2 and
§3.3 margin-flip numbers. The two runs are independent, so the numbers here are
provenance for the generations only, not for the reported margin-flip values.
