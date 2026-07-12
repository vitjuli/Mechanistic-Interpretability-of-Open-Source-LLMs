# Direction geometry results — §3.1–§3.5 and Appendices B, C, E

Per-layer geometry of the four directions (reading `w_LDA`, writing `Δμ`, use `u`, unembedding `γ`)
on the decay task and its variants, computed from the residual/gradient field dumps. Model: Qwen3-4B base.

## Core geometry
| file | what | thesis |
|---|---|---|
| `decodability_by_layer.csv` | held-out AUC per layer for `w_LDA` / `Δμ` / `u` | **Table 2** (`tab:decodability`), §3.1.1 |
| `axis_geometry_by_layer.csv` | per-layer cos(w,u)/cos(Δμ,u)/cos(u,γ)/cos(w,γ)/cos(Δμ,γ) + margin flip | **Table** `tab:cosines`, §3.1.3 |
| `coupling_curve.csv` | read/write/use on the residual stream, raw vs scaffold (36 layers × 2) | §3.1.3 / §2.6 |
| `lambda_shrinkage.csv` | Ledoit–Wolf λ, cos(w,u) robustness to shrinkage, σ per direction | §2.5.2 / §2.7 |
| `active_span_triad.csv` | active-feature span capture of each direction vs random null, per layer | **Table** `tab:capture`, §3.5 |
| `feature_delta_write_contribution.csv` | writer population by depth: n_writers 8→56, top-5 share 95%→70%, 91% of δ-write in L27–35 | §3.5 writer-population paragraph |
| `realized_decay.csv`, `realized_particles.csv`, `realized_scaffold.csv` | MLP share of rotation/length of `Δμ` + transcoder reconstruction relative error | **Table 5** (`tab:mlpshare`), §3.2 |

## Appendix B — decision zone
| file | what |
|---|---|
| `convergent_validation.json` | full-residual AUC (fisher/diffmeans/logreg) post-L21, used for the "linear classifier on the residual reaches 0.96" onset | `tab:decisionzone` |
| `axis_causality_summary.json` | held-out AUC 0.988–0.996 and per-measurement onsets | `tab:decisionzone` |

## Appendix C — spinning of directions
| file | what |
|---|---|
| `rotation_decay_by_concept.csv` | rotation decay ρ(Δk) of the three directions across 4 concepts (reading < writing < use) | `tab:spin` |

## Appendix E — whitening attenuation
`whitening/` — for both concepts (`_ab` = decay, `_gram` = grammar):
| file | what |
|---|---|
| `whitening_per_layer_{ab,gram}.csv` | per-layer cos(w,u), attenuation factor, prediction error | `tab:whitening`, F.1/F.2 |
| `whitening_numbers_{ab,gram}.json` | scalar results: attenuation, Rayleigh percentile, margin-identity ratio, eigen residual | F.2–F.4 |
| `whitening_counterfactual_{ab,gram}.csv` | margin-stratified rotation of w toward u vs prompt- and family-matched nulls | F.5 |
