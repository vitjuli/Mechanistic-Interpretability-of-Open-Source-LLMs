# Semantic steering bundle (L22 / L24 / L35)

Copied from `~/Downloads` on 2026-07-12 (source run dated 2026-07-11, generations timestamped 2026-07-12 01:51).
Origin: `colab/run_semantic_steering.sh`. Downloads artifacts were `semantic_out.tar.gz` + `semantic_steering_all.csv`.

## Contents
- `<pair>_L{22,24,35}.txt` — per-prompt steering logs: for each prompt, `base_margin` and the
  margin delta `dm` for the three directions (`delta` = Delta-mu writing, `usage` = u use direction,
  `w_res` = reading direction) across strengths c = 1,2,4,8,16,32, plus a 30-token generation.
- `semantic_steering_all.csv` — aggregate table across all pairs/layers (md5 341668d7...).

Pairs: `alpha_beta`, `electron_vs_{neutron,photon,proton}`, `neutron_vs_{photon,proton}`,
`photon_vs_proton`, `grammar`.

## Relation to the thesis
Plausible provenance for the §3.3 "Dissociation under intervention" generation tables
(particle steering at L24). NOTE these are a SEPARATE, later run from the aggregate per-pair CSVs
in the parent dir (`steering_pair_*.csv`), which were used for the §3.3 margin-flip verification.
Numbers here do not necessarily match those.

## 🔴 DIAGNOSED 2026-08-02 — this bundle is superseded, do not quote it

Two defects were established by reading the code and the corpora. The filenames are fine; the
run was not.

**1. Wrong contrast token (particles only).** `steering_decode_check.py` read the second margin
token from each prompt's `incorrect_answer` field. In the v2 particle corpora that field holds an
arbitrary distractor, not the partner class of the pair — most electron/neutron/photon prompts
carry `incorrect_answer = proton`. So `base_margin`, `dmargin` and `flipped` were measured against
a token no direction was ever steered toward. Per-prompt verdicts are in `label_audit.csv`
(`scripts/audit_semantic_steering_labels.py`): **64/128 rows usable** — `neutron_vs_proton`,
`alpha_beta`, `grammar` fully; `electron_vs_photon` and `electron_vs_proton` half;
`electron_vs_neutron`, `neutron_vs_photon`, `photon_vs_proton` not at all.

**2. Wrong checkpoint (all rows).** The script defaulted to `Qwen/Qwen3-4B-Base`, while the dumps
supplying the directions and σ are `Qwen/Qwen3-4B`. Directions from one model, generations from
another — this hits every row, including the ones the audit marks usable.

The push sign was **not** affected: `class_a` from the dump made it class-dependent for the
particle pairs, and `y_canonical` did the same for α/β and grammar.

Both defects are fixed in `scripts/steering_decode_check.py` (contrast = the corpus's own second
class, `--model` defaults to `Qwen/Qwen3-4B` and is asserted against `meta.npz`), and
`colab/run_semantic_steering.sh` now pins the checkpoint. App G takes numbers only from the
re-run, not from this bundle.

### Original symptom, kept for the record
The `correct=` header matches the filename's first token, but the `incorrect=` header (which sets
`base_margin` = margin of correct vs incorrect) does NOT match the filename's second token for most
particle pairs (L24 shown):

| file                        | internal label                 | filename implies |
|-----------------------------|--------------------------------|------------------|
| electron_vs_neutron_L24.txt | correct=electron incorrect=proton | ...neutron |
| electron_vs_photon_L24.txt  | correct=electron incorrect=proton | ...photon  |
| electron_vs_proton_L24.txt  | correct=electron incorrect=proton | ...proton ✓ |
| neutron_vs_photon_L24.txt   | correct=neutron incorrect=proton  | ...photon  |
| neutron_vs_proton_L24.txt   | correct=neutron incorrect=proton  | ...proton ✓ |
| photon_vs_proton_L24.txt    | correct=photon incorrect=electron | ...proton  |

Most pairs report `incorrect=proton` regardless of the filename. The cause is defect 1 above:
the label is not stale, it faithfully reports the (wrong) token the margin was read against.
