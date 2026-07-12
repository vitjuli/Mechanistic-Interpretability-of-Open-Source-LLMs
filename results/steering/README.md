# Steering results — §3.1.2 (causal inertness) and §3.3 (dissociation)

Outputs of the direction-steering scripts (`scripts/122_b1_steering_sweep.py`,
`scripts/85_steering_all_layers.py`, `scripts/steering_summary_table.py`,
`scripts/steering_decode_check.py`). Model: Qwen3-4B base, transcoders `mwhanna/qwen3-4b-transcoders`.

## §3.1.2 Causal inertness
| file | what | thesis |
|---|---|---|
| `steering_summary_all.csv` | margin flip + intact + cos(w,u)/cos(Δμ,u), all concepts (α/β vs particle pairs + probe) | Table `tab:geomcausal`, margin-flip claims |
| `steering_all_layers_curve.csv` | per-layer margin/intact curve, all 36 layers (decay probe) | reading-direction inert, beats null 0/36 layers |
| `steering_all_layers_summary.json` | per-layer summary of the above | same |

## §3.3 Dissociation under intervention
| file | what | thesis |
|---|---|---|
| `semantic_L22_L24_L35/semantic_steering_all.csv` | steering generations, all pairs at L22/L24/L35 | source of the generation Tables 6–7 |
| `semantic_L22_L24_L35/*.txt` | per-pair per-layer logs (base_margin + `delta/usage/w_res` margin deltas by c, plus 30-token generation) | supporting |
| `decode_e_p_L24.md` | electron/proton L24 decode: Δμ = coherent content, u = forced token with degenerate repetition | dissociation narrative |

⚠ `semantic_L22_L24_L35/README.md` documents a filename-vs-`incorrect=` label mismatch in that bundle —
read it before quoting any `base_margin` from those files.

## pairs/ — per-pair margin sweeps (both sections)
`steering_pair_{e_p,e_n,n_p,n_g,g_p,e_g_V2}.csv` and
`steering_{grammar,scaffold,rawsuffix,particles_eg_SCAFFOLD}.csv`: per-layer × strength × direction
margin flip / mean margin (columns `layer,c,dir,...,mean_dmargin_toward,margin_flip,...`). These back the
c = 8 dose-response sentence in §3.1.2 and the per-prompt margin cross-check of Tables 6–7 in §3.3.
