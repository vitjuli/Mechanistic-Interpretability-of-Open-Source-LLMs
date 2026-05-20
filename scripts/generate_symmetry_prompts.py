"""
Generate new prompts to make alpha/beta perfectly symmetric.

Current state (wording_variant=1 per group, all variants counted):
  L1: alpha=80, beta=80  ← already symmetric
  L2: alpha=88, beta=86  ← diff=2
  L3: alpha=40, beta=40  ← already symmetric
  AUX: alpha=28, beta=28 ← already symmetric
  Total: alpha=236, beta=234

Target after additions:
  L2: 116:116  (+28 alpha AP, +30 beta BP — BP3 has n=10 to absorb 2-diff)
  AUX: 33:33   (+5 AUX-AP alpha, +5 AUX-BP beta)
  Final: alpha=269, beta=269, total=538

New groups:
  ALPHA L2 (process level — closes LS-alpha3_proc gap):
    L2-AP1  n=10  cluster ejection (basic)            ↔ L2-BR2 (n→p)
    L2-AP2  n=10  cluster ejection + daughter Z/A     ↔ L2-BR7 (n→p + Z+1)
    L2-AP3  n=8   cluster ejection + no particle cre  ↔ L2-BR3 (n→p + antineutrino)

  BETA L2 (quark/mechanism level — genuinely new content):
    L2-BP1  n=10  quark flavour change (d→u)
    L2-BP2  n=10  W boson mediation
    L2-BP3  n=10  quark change + daughter Z+1  (n=10 not 8, absorbs 2-prompt AR/BR diff)

  ALPHA AUX (mechanism singles):
    AUX-AP1  no quark flavour change
    AUX-AP2  not weak-force mediated
    AUX-AP3  no new particles created
    AUX-AP4  discrete (monoenergetic) energy spectrum
    AUX-AP5  element shifts 2 positions back on periodic table

  BETA AUX (mechanism singles — genuinely new, no duplication of AUX-B1/B3/B5):
    AUX-BP1  continuous energy spectrum
    AUX-BP2  new particles created (e- and antineutrino)
    AUX-BP3  lepton number increases by 1
    AUX-BP4  emitted particle is lightest stable charged elementary particle
    AUX-BP5  element shifts 1 position forward on periodic table
"""

import json, os

OUT = os.path.join(os.path.dirname(__file__), '../data/prompts/physics_decay_type_probe_supplement.jsonl')
PREAMBLE = "A nucleus undergoes a decay process."
QUESTION  = "Is the decay type alpha or beta?"

def make(group_id, wording_variant, prompt_body,
         correct, level, level_label,
         cue_type=None, concept_route=None,
         relation_type=None, inference_steps=2,
         difficulty="medium", evidence="combination",
         has_alpha_kw=False, has_beta_kw=False,
         has_he=False, has_e=False,
         is_aux=False, abstraction_level=None,
         notes=""):
    ic = 0 if correct == " alpha" else 1
    incorr = " beta" if correct == " alpha" else " alpha"
    if abstraction_level is None:
        abstraction_level = inference_steps
    return {
        "prompt_id":   f"{group_id}-v{wording_variant:02d}",
        "prompt":      f"{PREAMBLE} {prompt_body} {QUESTION}",
        "correct_answer":   correct,
        "incorrect_answer": incorr,
        "behaviour":        "physics_decay_type_probe",
        "behaviour_type":   "latent_state_probing",
        "physics_concept":  "alpha_decay" if ic == 0 else "beta_decay",
        "concept_index":    ic,
        "template_idx":     wording_variant,
        "difficulty":       difficulty,
        "answer_format":    "x_or_y",
        "surface_family":   group_id,
        "keyword_free":     not (has_alpha_kw or has_beta_kw or has_he or has_e),
        "level":            level,
        "level_label":      level_label,
        "group_id":         group_id,
        "level_group":      group_id,
        "cue_type":         cue_type,
        "cue_set":          [cue_type] if cue_type else [],
        "relation_type":    relation_type,
        "concept_route":    concept_route,
        "wording_variant":  wording_variant,
        "test_type":        ["AUX"] if is_aux else ["SE", "IC"],
        "abstraction_level": abstraction_level,
        "inference_steps":  inference_steps,
        "has_alpha_keyword": has_alpha_kw,
        "has_beta_keyword":  has_beta_kw,
        "has_helium_ref":    has_he,
        "has_electron_ref":  has_e,
        "has_isotope":       False,
        "keyword_type":      None,
        "semantic_equivalence_group": group_id,
        "ic_concept_group":  "alpha" if ic == 0 else "beta",
        "contrastive_pair_id": None,
        "contrastive_role":    None,
        "is_anchor":   False,
        "is_kw_variant": False,
        "is_auxiliary":  is_aux,
        "prompt_format": "process",
        "evidence_completeness": evidence,
        "is_uniquely_determining": True,
        "notes": notes,
    }


records = []

# ── L2-AP1  (n=10)  cluster ejection — basic ─────────────────────────────
AP1 = [
    "A tightly bound cluster of two protons and two neutrons is ejected from the nucleus as a single unit.",
    "A preformed nuclear cluster consisting of two protons and two neutrons escapes the parent nucleus.",
    "Two protons and two neutrons spontaneously cluster together inside the nucleus and are expelled together.",
    "The nucleus ejects a bound grouping of four nucleons as a single unit.",
    "A cluster of four nucleons — two of each type — leaves the nucleus as one particle.",
    "A bound assembly of 2 protons and 2 neutrons tunnels out of the nucleus.",
    "The parent nucleus emits a cluster comprising 2 protons and 2 neutrons as a single entity.",
    "A compact nuclear cluster of four nucleons exits the nucleus in this decay.",
    "Two protons and two neutrons form a bound unit and escape the nucleus together.",
    "The decay proceeds by expulsion of a four-nucleon cluster held together by the nuclear force.",
]
for i, body in enumerate(AP1, start=1):
    records.append(make("L2-AP1", i, body, " alpha", 2, "relation_process",
                        cue_type="cluster_ejection", relation_type="cluster_formation",
                        inference_steps=3, difficulty="medium", evidence="combination",
                        notes="New: alpha process group — cluster ejection basic"))

# ── L2-AP2  (n=10)  cluster ejection + daughter Z/A ─────────────────────
AP2 = [
    "A cluster of 2 protons and 2 neutrons is ejected, leaving the daughter with Z−2 and A−4.",
    "A four-nucleon cluster escapes the nucleus; the daughter has atomic number 2 less and mass number 4 less than the parent.",
    "The nucleus expels a 2p2n cluster, causing the daughter to have two fewer protons and four fewer nucleons.",
    "A nuclear cluster of 4 nucleons is emitted, reducing the daughter's atomic number by 2 and mass number by 4.",
    "Two protons and two neutrons escape as a unit; the daughter retains A−4 nucleons and has Z−2.",
    "A bound cluster of 2 protons and 2 neutrons is emitted; the daughter has Z decreasing by 2 and A decreasing by 4.",
    "The ejected four-nucleon cluster leaves the daughter nucleus with 2 fewer protons and 4 fewer nucleons total.",
    "A 2p2n cluster escapes the nucleus, with the daughter having atomic number reduced by 2 and mass reduced by 4.",
    "Four nucleons (2 protons, 2 neutrons) are emitted together; the daughter's Z drops by 2 and A by 4.",
    "The emission of a bound 2p2n cluster results in a daughter nucleus with Z−2 and A−4 relative to the parent.",
]
for i, body in enumerate(AP2, start=1):
    records.append(make("L2-AP2", i, body, " alpha", 2, "relation_process",
                        cue_type="cluster_ejection_daughter", relation_type="cluster_formation_consequence",
                        inference_steps=3, difficulty="medium", evidence="combination",
                        notes="New: alpha process group — cluster ejection + daughter Z/A"))

# ── L2-AP3  (n=8)  cluster ejection + no particle creation ───────────────
AP3 = [
    "A cluster of existing nucleons is ejected from the nucleus; no new particles are created in this process.",
    "The decay involves emission of a pre-existing nuclear cluster; no particle creation events occur.",
    "Existing nucleons within the nucleus form a cluster and escape; the process involves no production of new particles.",
    "A bound cluster of 2 protons and 2 neutrons, already present in the nucleus, is expelled; no new particles are generated.",
    "The emitted particle consists entirely of nucleons already present in the parent nucleus; no particle creation takes place.",
    "The decay produces two objects: a daughter nucleus and an emitted cluster of pre-existing nucleons; no new particles appear.",
    "Two protons and two neutrons that were already part of the nucleus cluster together and escape; no transmutation of quarks occurs.",
    "The nucleus separates into a daughter nucleus and a cluster of pre-existing nucleons; no new particles are created during this process.",
]
for i, body in enumerate(AP3, start=1):
    records.append(make("L2-AP3", i, body, " alpha", 2, "relation_process",
                        cue_type="cluster_no_creation", relation_type="cluster_formation_process",
                        inference_steps=3, difficulty="hard", evidence="combination",
                        notes="New: alpha process group — cluster ejection + no particle creation"))

# ── L2-BP1  (n=10)  quark flavour change (d→u) ───────────────────────────
BP1 = [
    "A down quark inside a neutron transforms into an up quark, converting the neutron into a proton.",
    "Inside the nucleus, a quark undergoes a flavour change: a down quark becomes an up quark.",
    "A quark type transformation occurs: a down quark in a neutron is converted to an up quark.",
    "The decay involves the conversion of a down quark to an up quark within a neutron.",
    "A down-to-up quark transition occurs inside the nucleus, changing a neutron into a proton.",
    "The fundamental process is a quark flavour change: a d quark transforms into a u quark inside the neutron.",
    "Within a nucleon, a down quark transforms into an up quark, changing the nucleon's identity.",
    "A quark within a neutron undergoes flavour conversion from down to up, producing a proton.",
    "The decay is driven by a d→u quark transition inside a neutron, converting it to a proton.",
    "A down quark changes into an up quark inside the nucleus, converting a neutron into a proton.",
]
for i, body in enumerate(BP1, start=1):
    records.append(make("L2-BP1", i, body, " beta", 2, "relation_process",
                        cue_type="quark_flavour_change", relation_type="quark_level_process",
                        inference_steps=4, difficulty="hard", evidence="combination",
                        notes="New: beta quark-level process group — d→u quark transition"))

# ── L2-BP2  (n=10)  W boson mediation ────────────────────────────────────
BP2 = [
    "The transformation of a nucleon is mediated by the exchange of a virtual W⁻ boson.",
    "The decay proceeds via emission of a virtual W boson, which subsequently produces the observed particles.",
    "A virtual W⁻ boson acts as the intermediate particle enabling the nucleon transformation.",
    "The process involves a W boson as an intermediate state, mediating the neutron-to-proton conversion.",
    "Inside the nucleus, a virtual W⁻ boson is exchanged, driving the quark flavour change.",
    "The nucleon transformation is enabled by a short-lived virtual W⁻ boson intermediate.",
    "A W boson mediates the quark-level process that converts a neutron to a proton.",
    "The decay involves emission of a W⁻ boson, which then produces the observed decay products.",
    "A virtual W boson carries the interaction that changes a neutron into a proton in this decay.",
    "The process is mediated by a W⁻ boson, the force carrier of the weak nuclear interaction.",
]
for i, body in enumerate(BP2, start=1):
    records.append(make("L2-BP2", i, body, " beta", 2, "relation_process",
                        cue_type="w_boson_mediation", relation_type="weak_force_mechanism",
                        inference_steps=4, difficulty="hard", evidence="combination",
                        notes="New: beta mechanism group — W boson mediation"))

# ── L2-BP3  (n=10)  quark change + daughter Z+1  (n=10 absorbs 2-diff) ──
BP3 = [
    "A quark flavour change occurs inside a nucleon, increasing the atomic number of the daughter by 1.",
    "A down-to-up quark transition inside the nucleus increases the daughter's atomic number by 1.",
    "The conversion of a down quark to an up quark inside a neutron causes the daughter nucleus to have Z+1.",
    "A quark type change transforms a neutron to a proton, and the daughter's atomic number increases by one.",
    "The decay involves quark flavour conversion within a nucleon, resulting in a daughter with one more proton.",
    "Inside the nucleus, a d→u quark change occurs, converting a neutron to a proton and increasing Z by 1.",
    "A quark flavour transition within the nucleus causes a neutron to become a proton, raising the daughter's Z by 1.",
    "The fundamental quark transition (d→u) inside a nucleon leads to a daughter nucleus with Z+1.",
    "A down quark converts to an up quark inside the nucleus; the daughter acquires one extra proton relative to the parent.",
    "The d→u quark transition within a neutron produces a proton and a daughter nucleus with atomic number increased by 1.",
]
for i, body in enumerate(BP3, start=1):
    records.append(make("L2-BP3", i, body, " beta", 2, "relation_process",
                        cue_type="quark_change_z_plus1", relation_type="quark_level_consequence",
                        inference_steps=4, difficulty="hard", evidence="combination",
                        notes="New: beta quark + daughter group (n=10 to absorb 2-prompt AR/BR diff)"))

# ── AUX-AP  (n=1 each)  alpha mechanism singles ───────────────────────────
AUX_AP = [
    ("AUX-AP1", "no_quark_flavour_change",
     "No quark flavour change occurs in this decay.",
     "New: AUX alpha — no quark flavour change"),
    ("AUX-AP2", "not_weak_force",
     "The decay is not mediated by the weak nuclear force.",
     "New: AUX alpha — not weak force"),
    ("AUX-AP3", "no_new_particles",
     "No new fundamental particles are created in this decay; only existing nucleons are rearranged.",
     "New: AUX alpha — no new particles created"),
    ("AUX-AP4", "discrete_energy_spectrum",
     "The emitted particle carries a fixed, discrete kinetic energy (monoenergetic emission).",
     "New: AUX alpha — discrete energy spectrum (two-body decay)"),
    ("AUX-AP5", "element_shift_minus2",
     "After this decay, the resulting element is two positions earlier on the periodic table than the parent element.",
     "New: AUX alpha — element shifts back 2 on periodic table"),
]
for gid, cue, body, note in AUX_AP:
    records.append(make(gid, 1, body, " alpha", None, "auxiliary",
                        cue_type=cue, inference_steps=5, difficulty="hard",
                        abstraction_level=5, evidence="single",
                        is_aux=True, notes=note))

# ── AUX-BP  (n=1 each)  beta mechanism singles ────────────────────────────
AUX_BP = [
    ("AUX-BP1", "continuous_energy_spectrum",
     "The emitted particle has a continuous energy spectrum rather than a discrete energy value.",
     "New: AUX beta — continuous energy spectrum (three-body decay)"),
    ("AUX-BP2", "new_particles_created",
     "New particles are created during this decay: the emitted particle and an antineutrino are produced from the decay energy.",
     "New: AUX beta — particle creation (e- and antineutrino created)"),
    ("AUX-BP3", "lepton_number_increases",
     "The total lepton number of the system increases by 1 in this decay.",
     "New: AUX beta — lepton number increases"),
    ("AUX-BP4", "lightest_charged_elementary",
     "The emitted particle is the lightest stable charged elementary particle known.",
     "New: AUX beta — emitted particle = lightest stable charged elementary (electron)"),
    ("AUX-BP5", "element_shift_plus1",
     "After this decay, the resulting element is one position later on the periodic table than the parent element.",
     "New: AUX beta — element shifts forward 1 on periodic table"),
]
for gid, cue, body, note in AUX_BP:
    records.append(make(gid, 1, body, " beta", None, "auxiliary",
                        cue_type=cue, inference_steps=5, difficulty="hard",
                        abstraction_level=5, evidence="single",
                        is_aux=True, notes=note))


# ── Verify counts ─────────────────────────────────────────────────────────
from collections import Counter
alpha_by_level = Counter()
beta_by_level  = Counter()
for r in records:
    lv = r['level_label']
    if r['correct_answer'] == ' alpha':
        alpha_by_level[lv] += 1
    else:
        beta_by_level[lv] += 1

print("=== NEW PROMPTS GENERATED ===")
print(f"Total new: {len(records)}")
print(f"  Alpha new L2:  {alpha_by_level['relation_process']:3d}")
print(f"  Beta  new L2:  {beta_by_level['relation_process']:3d}")
print(f"  Alpha new AUX: {alpha_by_level['auxiliary']:3d}")
print(f"  Beta  new AUX: {beta_by_level['auxiliary']:3d}")
print()
print("=== PROJECTED FINAL CORPUS ===")
print(f"  L1:  alpha= 80  beta= 80  (unchanged)")
print(f"  L2:  alpha={88+alpha_by_level['relation_process']:3d}  beta={86+beta_by_level['relation_process']:3d}  {'✓' if 88+alpha_by_level['relation_process']==86+beta_by_level['relation_process'] else 'MISMATCH'}")
print(f"  L3:  alpha= 40  beta= 40  (unchanged)")
print(f"  AUX: alpha={28+alpha_by_level['auxiliary']:3d}  beta={28+beta_by_level['auxiliary']:3d}  {'✓' if 28+alpha_by_level['auxiliary']==28+beta_by_level['auxiliary'] else 'MISMATCH'}")
a_total = 236 + alpha_by_level['relation_process'] + alpha_by_level['auxiliary']
b_total = 234 + beta_by_level['relation_process'] + beta_by_level['auxiliary']
print(f"  TOTAL: alpha={a_total}  beta={b_total}  {'✓ SYMMETRIC' if a_total==b_total else 'MISMATCH'}")
print()

# Write JSONL
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"Written: {OUT}")
print(f"Lines: {len(records)}")

# ── Per-group summary ─────────────────────────────────────────────────────
print("\n=== NEW GROUPS SUMMARY ===")
groups = {}
for r in records:
    g = r['group_id']
    if g not in groups:
        groups[g] = {'n': 0, 'ans': r['correct_answer'].strip(), 'level': r['level_label'], 'cue': r['cue_type']}
    groups[g]['n'] += 1
for g, info in sorted(groups.items()):
    print(f"  {g:12s}  n={info['n']:2d}  ans={info['ans']:5s}  level={info['level']:18s}  cue={info['cue']}")
