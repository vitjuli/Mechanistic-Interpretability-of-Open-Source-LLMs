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

## ⚠️ Known issue — filename vs internal `incorrect=` label mismatch
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

Most pairs report `incorrect=proton` regardless of the filename. Before using any base_margin / dm
value from this bundle in the thesis (in particular Table 7 = electron_vs_photon), confirm which
incorrect class the run actually steered against — the header label may be stale/buggy, or the
filenames may be wrong. Do NOT trust the pairing on filename alone.
