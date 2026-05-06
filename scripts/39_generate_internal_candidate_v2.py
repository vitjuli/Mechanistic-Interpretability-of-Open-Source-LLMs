"""
Generate physics_internal_candidate_selection_v2 (~500 prompts).

Strategy B: Clean internal-only.
  - 6 single-token filters × ~77 prompts ≈ 447 (main corpus, placed first)
  - 2 multi-token filters × 20 prompts = 40  (generalisation appendix, placed last)
  - Total: ~487

Multi-token prompts are placed LAST so `--n_prompts 447` in scripts 06/07
naturally excludes them without any additional filtering.

Removed vs mini:
  - beta_minus_charged_product (61.5%): normalization artifact + genuine ambiguity
  - atomic_number_carrier (77.8%): under-constrained jargon context

Redesigned:
  - massless_em_mediator: only 6 F1 prompts; rest in F2/F3/F4

Filters:
  1-token: negative_charge_atomic→electron, positive_charge_atomic→proton,
           neutral_charge_atomic→neutron, nuclear_stability_provider→neutron,
           em_quantum→photon, massless_em_mediator→photon
  2-token: antielectron→positron (generalisation), heavy_negative_lepton→muon (generalisation)

Token audit:
  electron  OK (1-tok)    proton   OK (1-tok)
  neutron   OK (1-tok)    photon   OK (1-tok)
  positron  MULTI (2 tok) — placed at end, skipped by script 07
  muon      MULTI (2 tok) — placed at end, skipped by script 07
"""

import json
import random
from pathlib import Path

random.seed(42)

BEHAVIOUR = "physics_internal_candidate_selection_v2"
OUT_PATH = Path("data/prompts") / f"{BEHAVIOUR}_train.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

PRIO = {
    "electron":  [" positron", " muon",    " proton",   " neutron", " photon"],
    "photon":    [" electron", " muon",    " positron", " proton",  " neutron"],
    "neutron":   [" proton",   " positron"," electron", " muon",    " photon"],
    "positron":  [" electron", " muon",    " proton",   " neutron", " photon"],
    "muon":      [" positron", " electron"," proton",   " neutron", " photon"],
    "proton":    [" neutron",  " electron"," positron", " muon",    " photon"],
}

def s(p): return p.strip()

def pinc(pool, correct):
    wrong = [p for p in pool if p != correct]
    for pref in PRIO.get(s(correct), []):
        if pref in wrong:
            return pref
    return wrong[0]

def R(prompt, correct, pool, fp, wf, exp="core", exp_internal=None, multi_token=False):
    wrong = [p for p in pool if p != correct]
    inc   = pinc(pool, correct)
    return {
        "prompt": prompt,
        "correct_answer": correct,
        "incorrect_answer": inc,
        "incorrect_answers": wrong,
        "implicit_candidate_pool": [s(p) for p in pool],
        "expected_internal_candidates": exp_internal or [s(p) for p in pool],
        "target_candidate": s(correct),
        "filter_property": fp,
        "wording_family": wf,
        "candidate_type": "internal_particle_candidate",
        "explicit_candidate_set": False,
        "experiment_type": exp,
        "multi_token_answer": multi_token,
        "filter_correct_id": f"{fp}__{s(correct)}",
    }

def make_records(cfg):
    exp_type  = cfg.get("exp_type", "core")
    correct   = cfg["correct"]
    pool      = cfg["pool"]
    fp        = cfg["fp"]
    exp_int   = cfg["exp_internal"]
    mt        = cfg.get("multi_token", False)
    recs = []
    for prompt in cfg.get("f1", []):
        recs.append(R(prompt, correct, pool, fp, "F1_direct_implicit",
                      exp=exp_type, exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f2", []):
        recs.append(R(prompt, correct, pool, fp, "F2_contextual_implicit",
                      exp=exp_type, exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f3", []):
        recs.append(R(prompt, correct, pool, fp, "F3_process_implicit",
                      exp=exp_type, exp_internal=exp_int, multi_token=mt))
    for prompt in cfg.get("f4", []):
        recs.append(R(prompt, correct, pool, fp, "F4_contrast_implicit",
                      exp=exp_type, exp_internal=exp_int, multi_token=mt))
    return recs

# ── filter configurations ─────────────────────────────────────────────────────

CFGS_SINGLE = [

    # ── 1. negative_charge_atomic → electron (10+25+25+20 = 80) ──────────────
    dict(
        fp="negative_charge_atomic", correct=" electron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False, exp_type="core",
        f1=[
            "Which particle in an atom carries negative electric charge? Answer:",
            "Which subatomic particle has a charge of −1 elementary units? Answer:",
            "Which atomic particle orbits the nucleus and is negatively charged? Answer:",
            "Which particle found outside the atomic nucleus is negatively charged? Answer:",
            "What is the negatively charged particle that surrounds the atomic nucleus? Answer:",
            "Which fundamental atomic particle has a charge of −e? Answer:",
            "Which particle determines the chemical properties of an atom through its negative charge? Answer:",
            "Which constituent of an atom carries a single unit of negative electrical charge? Answer:",
            "Which atomic particle has charge −1 and minimal mass compared to nuclear particles? Answer:",
            "Which negatively charged particle is the basic unit of electric current in metals? Answer:",
        ],
        f2=[
            "In the Bohr model of the atom, particles orbit the nucleus in quantized energy levels. The orbiting particles carry negative charge. What are they called? Answer:",
            "Chemical bonds between atoms form when tiny negatively charged particles are shared or transferred between nuclei. Which particle is transferred or shared? Answer:",
            "Electricity flows through copper wire because negatively charged particles drift through the metal lattice. What are these charge carriers? Answer:",
            "When atoms gain extra negatively charged particles, they become anions. Which particles are added to form anions? Answer:",
            "In Thomson's cathode ray experiment, a beam of negative particles travelled from cathode to anode in a vacuum tube. What were these particles? Answer:",
            "The outermost shell of an atom determines its reactivity and is filled with negatively charged particles. What are these particles? Answer:",
            "In semiconductors, n-type doping introduces extra negatively charged carriers into the conduction band. What are these carriers? Answer:",
            "In quantum mechanics, atomic orbitals (s, p, d, f) describe probability distributions of negatively charged particles around the nucleus. Which particles occupy these orbitals? Answer:",
            "The photoelectric effect ejects negatively charged particles from a metal surface when light above a threshold frequency shines on it. What particles are ejected? Answer:",
            "A neutral hydrogen atom consists of one positive proton in the nucleus and one negatively charged particle in orbit. What is this orbiting particle? Answer:",
            "Atomic emission spectra arise when negatively charged particles fall from higher to lower energy levels, releasing photons. Which particles undergo these transitions? Answer:",
            "Ionic compounds form when one atom donates negatively charged particles to another, creating oppositely charged ions. What particles are donated? Answer:",
            "In electronegativity, certain atoms pull the shared negatively charged bonding particles closer to themselves. What are these particles? Answer:",
            "Ionisation energy is the energy required to remove a negatively charged particle from a neutral atom. Which particle is removed? Answer:",
            "Covalent bonds form when two atoms share pairs of negatively charged particles between their nuclei. What particles are shared? Answer:",
            "In a neon sign, an electric field accelerates negatively charged particles through neon gas, which then emits coloured light. What are these particles? Answer:",
            "The plum-pudding model of the atom proposed negatively charged particles embedded in a diffuse positive sphere. What were these particles? Answer:",
            "Valence shells determine how atoms bond because they contain the outermost negatively charged particles. What are these particles? Answer:",
            "In scanning tunnelling microscopy, a tiny current of negatively charged particles tunnels between the probe and sample surface. What particles tunnel? Answer:",
            "During electrophoresis, negatively charged particles in molecules migrate toward the positive electrode under an applied voltage. Which particles drive this migration? Answer:",
            "In transistors, the gate voltage controls flow of negatively charged particles through the semiconductor channel. What particles flow? Answer:",
            "In the hydrogen atom, the ground state energy is −13.6 eV, which is the binding energy of its single negatively charged particle. What particle is bound? Answer:",
            "Rydberg atoms have their outermost negatively charged particle in a very high energy level far from the nucleus. Which particle is in the Rydberg state? Answer:",
            "In X-ray fluorescence, a high-energy photon ejects a negatively charged inner-shell particle, causing outer-shell particles to fill the gap. What particles are involved? Answer:",
            "In photoemission spectroscopy, light ejects negatively charged particles from a material and their energies reveal the electronic structure. What particles are ejected? Answer:",
        ],
        f3=[
            "In ionic bonding, one atom transfers its outermost negatively charged particles to another, forming positive and negative ions. What particles are transferred? Answer:",
            "During electrolysis of water, negatively charged particles flow through the external circuit while ions move through the electrolyte. What particles flow externally? Answer:",
            "When ultraviolet light hits a zinc plate, negatively charged particles are ejected from the surface in the photoelectric effect. Which particles are released? Answer:",
            "X-rays are produced when fast negatively charged particles decelerate upon striking a metal target, emitting bremsstrahlung radiation. What are these particles? Answer:",
            "In thermionic emission, heating a cathode causes negatively charged particles to escape from the metal surface. What particles are emitted? Answer:",
            "Geiger-Müller counters detect ionising radiation by counting negatively charged particles freed from gas molecules by incoming radiation. What particles are counted? Answer:",
            "When an atom absorbs a photon, one of its negatively charged particles jumps to a higher energy level. Which particle absorbs the energy? Answer:",
            "In pair production near a nucleus, a high-energy photon creates a negatively charged particle and its antiparticle. What is the negatively charged particle? Answer:",
            "Electrostatic induction redistributes negatively charged particles within a conductor when an external charge is brought nearby. What particles redistribute? Answer:",
            "In a cloud chamber, supersaturated vapour condenses along the tracks of negatively charged particles. What particles leave visible tracks? Answer:",
            "Lightning discharges when negatively charged particles accumulate in storm clouds and suddenly flow to the ground. What particles flow? Answer:",
            "In Compton scattering, a photon transfers some of its momentum to a negatively charged particle, causing both to scatter. What particle is struck? Answer:",
            "During bremsstrahlung, negatively charged particles decelerate in the electric field of a nucleus and radiate photons. What particles decelerate? Answer:",
            "In electron capture, a nucleus absorbs a negatively charged particle from its innermost orbital shell. What particle is absorbed? Answer:",
            "Auger emission occurs when a core vacancy is filled and the released energy ejects a negatively charged particle from the atom. What particle is ejected? Answer:",
            "In beta-minus radioactive decay, a neutron emits a negatively charged particle and an antineutrino. What is the emitted particle? Answer:",
            "During oxidation in chemistry, a substance loses negatively charged particles. What particles are lost? Answer:",
            "In photoionisation, a photon delivers sufficient energy to remove a negatively charged particle from an atom. What particle is removed? Answer:",
            "Linear accelerators accelerate beams of negatively charged particles to near light speed for physics experiments. What particles are accelerated? Answer:",
            "During reduction in electrochemistry, a substance gains negatively charged particles at the cathode. What particles are gained? Answer:",
            "In laser cooling, photons are absorbed and re-emitted to slow down negatively charged particles in a trap. What particles are trapped? Answer:",
            "In photomultiplier tubes, incoming photons release negatively charged particles that are amplified by a cascade of dynodes. What particles are released initially? Answer:",
            "In an oscilloscope's cathode ray tube, a beam of negatively charged particles is steered by electric and magnetic fields. What particles form the beam? Answer:",
            "In positron emission tomography, an emitted positron immediately annihilates with a nearby negatively charged atomic particle. What particle annihilates with the positron? Answer:",
            "In field emission microscopy, a strong electric field extracts negatively charged particles from a sharp metal tip. What particles are extracted? Answer:",
        ],
        f4=[
            "Unlike the proton, which particle in the atom carries negative electric charge? Answer:",
            "Unlike the neutron, which atomic particle is electrically charged and negative? Answer:",
            "Unlike the positively charged proton, which atomic particle carries charge −1? Answer:",
            "While the proton sits inside the nucleus, which negatively charged particle occupies the outer shells? Answer:",
            "Unlike the neutral neutron, which atomic particle has a negative electric charge? Answer:",
            "The nucleus contains protons and neutrons; outside the nucleus, which negatively charged particle orbits? Answer:",
            "Unlike massive nuclear particles, which light, negatively charged lepton orbits around the nucleus? Answer:",
            "Unlike the proton (charge +1) and neutron (charge 0), which atomic particle has charge −1? Answer:",
            "Unlike the positron (antimatter), which stable, negatively charged particle surrounds the atomic nucleus? Answer:",
            "Protons are confined to the nucleus; which negatively charged particle exists in the region outside the nucleus? Answer:",
            "While the nucleus is overall positively charged, the outer shells contain which negatively charged particle that balances the nuclear charge? Answer:",
            "Unlike the neutral photon, which massive atomic particle carries negative electric charge? Answer:",
            "Unlike quarks which are confined inside protons and neutrons, which free negatively charged particle orbits the nucleus? Answer:",
            "Unlike the alpha particle (charge +2), the beta-minus particle has negative charge. What is the beta-minus particle? Answer:",
            "While neutrons contribute mass without charge, the particle that balances proton charge in the neutral atom is negatively charged. What is it? Answer:",
            "Positive ions form when an atom loses its outermost negatively charged particle. Which particle is lost? Answer:",
            "In contrast to the proton which repels other protons electromagnetically, which particle is attracted to the proton via the electric force? Answer:",
            "Unlike the muon (which is unstable and heavier), which stable, negatively charged particle is the lightest charged lepton? Answer:",
            "While photons are massless and neutral, atomic structure requires negatively charged massive particles to orbit the nucleus. What are these? Answer:",
            "Unlike the W− boson (which mediates weak charged current interactions), which stable negatively charged particle is a constituent of all atoms? Answer:",
        ],
    ),

    # ── 2. positive_charge_atomic → proton (10+25+25+20 = 80) ────────────────
    dict(
        fp="positive_charge_atomic", correct=" proton",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False, exp_type="core",
        f1=[
            "Which particle in the atomic nucleus carries positive electric charge? Answer:",
            "Which positively charged particle is found in every atomic nucleus? Answer:",
            "Which nuclear particle carries a single unit of positive charge? Answer:",
            "Which particle in the nucleus has a charge of +1 elementary unit? Answer:",
            "Which subatomic particle resides in the nucleus and carries positive charge? Answer:",
            "Which nuclear particle, when counted, gives the atomic number of an element? Answer:",
            "Which positively charged constituent of the nucleus determines which element an atom belongs to? Answer:",
            "Which nuclear particle has charge +e? Answer:",
            "Which massive particle in the nucleus is positively charged? Answer:",
            "Which particle, found in every nucleus, balances the negative charge of orbiting electrons in a neutral atom? Answer:",
        ],
        f2=[
            "The nucleus of a hydrogen atom consists of a single positively charged particle. What is this particle? Answer:",
            "Rutherford's gold foil experiment revealed a small, dense, positively charged nucleus. This positive charge comes from which particle? Answer:",
            "The periodic table organises elements by atomic number, which counts a specific positively charged particle in the nucleus. What is this particle? Answer:",
            "Nuclear charge determines which element an atom is; the positive charge comes from specific particles in the nucleus. What are they? Answer:",
            "Moseley showed that atomic number equals the number of positively charged particles in the nucleus. Which particles are counted? Answer:",
            "In stellar nucleosynthesis, the sun fuses hydrogen nuclei; each hydrogen nucleus is a single positively charged particle. What is it? Answer:",
            "Alpha particles consist of two positively charged nuclear particles and two neutrons. What are the positively charged components? Answer:",
            "The atomic number of an element equals the count of positively charged particles in its nucleus. What particles are counted? Answer:",
            "In nuclear physics, two of these positively charged particles in a helium-4 nucleus repel each other electromagnetically but are held together by the strong force. What particles? Answer:",
            "The Large Hadron Collider accelerates beams of positively charged nuclear particles to near the speed of light. What particles are accelerated? Answer:",
            "Proton therapy for cancer treatment directs beams of positively charged nuclear particles at tumours. What are these particles? Answer:",
            "In Bohr's atomic model, the positively charged nucleus attracts orbiting electrons. The nuclear positive charge comes from which particles? Answer:",
            "Nuclear binding energy holds positively charged particles together in the nucleus against their mutual electrostatic repulsion. What are these particles? Answer:",
            "In mass spectrometry, ions are separated by charge-to-mass ratio; the positive charge comes from the nucleus's positively charged particles. What are these? Answer:",
            "When a nucleus undergoes transmutation by gaining or losing positively charged particles, the element changes. What particles change? Answer:",
            "In a hydrogen discharge tube, removing the single outer electron leaves a bare positively charged nucleus. This nucleus is a single what? Answer:",
            "In the early universe, free positively charged particles combined with electrons to form the first neutral hydrogen atoms. What were these particles? Answer:",
            "Nuclear magnetic resonance (NMR) uses the spin of positively charged particles in nuclei to image chemical structures. Which particles precess? Answer:",
            "In nuclear fission, a uranium nucleus splits into two smaller nuclei; both contain positively charged particles distributed between them. What are these particles? Answer:",
            "Synchrotrons accelerate beams of positively charged nuclear particles to nearly the speed of light for collision experiments. What particles are in these beams? Answer:",
            "In cyclotron accelerators used in medicine, positively charged nuclear particles are accelerated by alternating electric fields. What are these particles? Answer:",
            "In the quark model, each of these positively charged nuclear particles contains two up quarks and one down quark. At the nuclear level, this particle is called what? Answer:",
            "Charged particle tracks in a bubble chamber curve in a magnetic field; positive curvature identifies which stable nuclear particle? Answer:",
            "In Millikan's oil drop experiment, the fundamental unit of positive charge was identified as the charge carried by which nuclear particle? Answer:",
            "In electrochemistry, the oxidation state of a metal is determined by the count of positively charged nuclear particles in its nucleus. What particles are counted? Answer:",
        ],
        f3=[
            "In beta-minus decay, a neutron in the nucleus transforms into a positively charged particle while emitting an electron and antineutrino. What is this positively charged product? Answer:",
            "In nuclear fusion, light nuclei combine by overcoming the repulsion between their positively charged particles. What are these particles? Answer:",
            "In the proton-proton chain in stellar cores, two positively charged nuclear particles fuse to begin helium synthesis. What are these particles? Answer:",
            "When a nucleus undergoes radioactive alpha decay, it loses two positively charged nuclear particles along with two neutrons. What are these particles? Answer:",
            "In Rutherford scattering, alpha particles are deflected by the concentrated positive charges of nuclear particles. What particles cause the deflection? Answer:",
            "During electron capture, a nucleus absorbs an orbital electron and one of its positively charged particles converts to a neutron. What was the positively charged particle? Answer:",
            "In nuclear spallation, high-energy neutrons knock positively charged particles out of nuclei. What particles are knocked out? Answer:",
            "In inverse beta decay, an antineutrino interacts with a positively charged nuclear particle and converts it to a neutron. What was the starting particle? Answer:",
            "In the CNO cycle in massive stars, carbon catalyses the fusion of positively charged particles to form helium. What particles are fused? Answer:",
            "During muon-catalysed fusion, muons bring positively charged particles close enough to fuse. What particles undergo fusion? Answer:",
            "In deep inelastic scattering experiments, high-energy electrons probe the internal structure of positively charged nuclear particles. What particles are probed? Answer:",
            "In hot big bang nucleosynthesis, free positively charged particles combined with neutrons to form helium nuclei. What were these particles? Answer:",
            "In linear accelerators (linacs), positively charged nuclear particles are accelerated through a series of hollow drift tubes. What are these particles? Answer:",
            "In nuclear transmutation using a particle accelerator, bombarding a target nucleus with positively charged particles can change the element. What particles are used as projectiles? Answer:",
            "When two hydrogen-1 atoms collide and fuse in stellar interiors, their two positively charged nuclei merge. Each nucleus is a single what? Answer:",
            "In particle physics detectors, the lightest stable positive track in a magnetic field identifies which nuclear particle? Answer:",
            "In Rutherford's model, alpha particles scatter from concentrations of positive charge in the nucleus. These charges come from which particles? Answer:",
            "In proton exchange membrane fuel cells, positively charged nuclear particles migrate through the membrane from anode to cathode. What particles migrate? Answer:",
            "In a fission reactor, the chain reaction produces daughter nuclei identified by their count of positively charged particles. What particles are counted? Answer:",
            "During hot plasma confinement in a tokamak, positively charged nuclear particles from hydrogen isotopes must be held at high temperature. What particles are confined? Answer:",
            "In high-energy particle colliders, beams of positively charged nuclear particles collide with antiprotons to produce new particles. What particles form the beams? Answer:",
            "In radiocarbon dating, the isotope carbon-14 decays to nitrogen-14, gaining a positively charged nuclear particle in the process. What particle is gained? Answer:",
            "In nuclear activation analysis, the identity of elements is determined by their count of positively charged nuclear particles. What particles are counted? Answer:",
            "In Bremsstrahlung production at an X-ray tube, fast electrons decelerate near the positively charged nuclear particles of a tungsten target. What nuclear particles attract the electrons? Answer:",
            "In laser-driven inertial confinement fusion, powerful lasers compress a fuel pellet until its positively charged nuclei fuse. What positively charged particles fuse? Answer:",
        ],
        f4=[
            "Unlike the neutron, which particle in the atomic nucleus carries positive electric charge? Answer:",
            "Unlike the electron, which nuclear particle has positive charge? Answer:",
            "Unlike the electrically neutral neutron, which nuclear particle has a charge of +1? Answer:",
            "While electrons orbit outside the nucleus, which positively charged particle is confined to the nucleus? Answer:",
            "Unlike the neutron (charge 0) and electron (charge −1), which nuclear particle has charge +1? Answer:",
            "Unlike the massless photon, which massive, positively charged nuclear particle is found in all nuclei? Answer:",
            "While neutrons add mass without charge, which nuclear particle adds both mass and positive charge to the nucleus? Answer:",
            "Unlike the negatively charged beta particle (electron), which particle remains in the nucleus and has positive charge? Answer:",
            "While neutrons can change by beta decay without changing the element, which positively charged particle defines the element? Answer:",
            "Unlike the photon which mediates forces, which positively charged matter particle is found in every atomic nucleus? Answer:",
            "Unlike positrons (which are antimatter and annihilate rapidly), which stable positively charged particle lives in every ordinary nucleus? Answer:",
            "Unlike the muon (a lepton), which stable, positively charged baryon constitutes the nucleus alongside neutrons? Answer:",
            "While the electron can be shared in covalent bonds, which positively charged nuclear particle stays in the nucleus in normal chemistry? Answer:",
            "Unlike alpha particles (charge +2), which single positively charged nuclear particle has charge +1? Answer:",
            "Unlike the antiproton (which annihilates with protons), which stable positively charged nuclear particle exists in ordinary matter? Answer:",
            "Unlike the W+ boson (which mediates weak charged current interactions), which stable positively charged nuclear particle has charge +1 and long lifetime? Answer:",
            "Neutrons add to atomic mass without changing the element; which positively charged particle, when added, changes the element itself? Answer:",
            "Unlike muons which decay in microseconds, which stable positively charged particle persists indefinitely in atomic nuclei? Answer:",
            "While electrons are leptons and neutrons are neutral baryons, which nuclear particle is a positively charged baryon? Answer:",
            "Unlike the pion (a meson that mediates residual nuclear force), which stable positively charged particle is a permanent constituent of nuclei? Answer:",
        ],
    ),

    # ── 3. neutral_charge_atomic → neutron (10+25+20+20 = 75) ────────────────
    dict(
        fp="neutral_charge_atomic", correct=" neutron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["electron", "proton", "neutron"],
        multi_token=False, exp_type="core",
        f1=[
            "Which particle in the atomic nucleus has no electric charge? Answer:",
            "Which nuclear particle is electrically neutral? Answer:",
            "Which particle in the nucleus carries zero electric charge? Answer:",
            "Which uncharged particle makes up atomic nuclei alongside protons? Answer:",
            "Which nuclear particle has zero charge but significant mass? Answer:",
            "Which nucleon is neither positively nor negatively charged? Answer:",
            "Which constituent of atomic nuclei carries no electric charge? Answer:",
            "Which particle contributes to atomic mass but not to atomic charge? Answer:",
            "Which particle in the nucleus is neutral? Answer:",
            "Which neutral particle, together with protons, forms the atomic nucleus? Answer:",
        ],
        f2=[
            "The atomic nucleus contains positively charged particles and electrically neutral particles. The neutral ones contribute to mass but not charge. What are they? Answer:",
            "Isotopes of an element have the same atomic number but different mass numbers because they differ in the count of neutral nuclear particles. What are these neutral particles? Answer:",
            "In nuclear physics, two types of particles make up the nucleus: protons carrying positive charge, and neutral particles that carry no charge at all. What are the neutral ones called? Answer:",
            "The mass number of a nucleus equals the total count of its protons plus its neutral nuclear particles. What are these neutral particles? Answer:",
            "Deuterium, the heavy isotope of hydrogen, contains one proton and one neutral nuclear particle. What is this neutral particle? Answer:",
            "Carbon-12 has six protons and six neutral nuclear particles in its nucleus. What are these neutral particles? Answer:",
            "James Chadwick discovered a neutral nuclear particle in 1932 by bombarding beryllium with alpha particles and detecting the recoil of protons. What did he discover? Answer:",
            "Nuclear stability requires that the number of neutral nuclear particles increases faster than proton number in heavy elements. What are these neutral particles? Answer:",
            "In lead-208, there are 82 protons and 126 neutral nuclear particles. What are these neutral particles? Answer:",
            "In a fission reactor, the chain reaction is initiated by absorption of a slow neutral nuclear particle. What particle is absorbed? Answer:",
            "The nuclear strong force acts equally between protons and the neutral particles in the nucleus. What are these neutral particles? Answer:",
            "Boron control rods in nuclear reactors absorb neutral particles to regulate the fission rate. What particles are absorbed? Answer:",
            "Heavy water (D2O) contains deuterium nuclei, each with one proton and one neutral nuclear particle. What is the neutral particle? Answer:",
            "In neutron diffraction, a beam of neutral particles probes the crystal structure of materials without electromagnetic interference. What are these neutral particles? Answer:",
            "In stellar nucleosynthesis, rapid neutron capture (r-process) involves a nucleus absorbing neutral particles faster than it can decay. What particles are captured? Answer:",
            "In nuclear transmutation experiments, bombarding a nucleus with neutral particles can produce new isotopes without changing the element's charge. What particles are used? Answer:",
            "In activation analysis, materials are irradiated with neutral nuclear particles to make them temporarily radioactive for identification. What particles are used? Answer:",
            "Moderated reactors use water or graphite to slow down neutral particles emitted during fission so they can trigger further fission. What particles are moderated? Answer:",
            "In a nuclear bomb, a supercritical mass allows neutral particles from fission to trigger a chain reaction. What particles propagate the chain? Answer:",
            "In cold neutron facilities, very slow neutral particles probe the structure of soft matter and biological molecules. What are these neutral particles? Answer:",
            "Uranium-238 undergoes neutron capture to produce plutonium-239; the neutral particles responsible for this transmutation are what? Answer:",
            "In nuclear magnetic resonance of deuterium, the neutral particle in the deuterium nucleus contributes to the nuclear spin signal. What particle is this? Answer:",
            "In spallation neutron sources, high-energy protons bombard a heavy metal target to produce beams of neutral nuclear particles. What are these particles? Answer:",
            "In muon-catalysed fusion of deuterium, the neutral particles in each deuterium nucleus are brought close enough to fuse. What neutral particles fuse? Answer:",
            "In the valley of stability, nuclei with too many neutral particles relative to protons undergo beta-minus decay. What are these excess neutral particles? Answer:",
        ],
        f3=[
            "In beta-minus decay, a neutral nuclear particle in the nucleus transforms into a proton, emitting an electron and antineutrino. What was the original particle? Answer:",
            "In nuclear fission, uranium-235 absorbs a slow neutral particle and splits into two smaller nuclei. What particle triggers the fission? Answer:",
            "Radioactive isotopes with too many neutral nuclear particles decay by beta-minus emission, converting a neutral particle to a proton. What particle converts? Answer:",
            "In nuclear fusion in the sun, deuterium nuclei each containing a neutral particle fuse together. What neutral particle is part of each deuterium? Answer:",
            "During nuclear transmutation in a reactor, a nucleus captures a neutral particle and becomes a heavier isotope of the same element. What particle is captured? Answer:",
            "In neutron activation, bombarding a stable nucleus with a neutral particle makes it radioactive without changing the element. What particle is the projectile? Answer:",
            "In the r-process (rapid neutron capture), neutral nuclear particles are absorbed faster than beta decay can occur in neutron star mergers. What particles are captured rapidly? Answer:",
            "When a nuclear reactor operates, fission releases fast neutral particles that must be slowed by a moderator to sustain the chain. What particles are moderated? Answer:",
            "In prompt fission, several fast neutral particles are released per fission event and go on to trigger further fissions. What neutral particles are released? Answer:",
            "In boron neutron capture therapy, neutral particles are captured by boron atoms adjacent to tumour cells. What particles are captured? Answer:",
            "During stellar s-process nucleosynthesis, nuclei accumulate neutral particles one at a time, building heavier stable isotopes. What particles are captured? Answer:",
            "In elastic scattering experiments, a beam of neutral nuclear particles bounces off nuclei without being absorbed, probing nuclear size. What neutral particles scatter? Answer:",
            "In beta-plus decay, a proton converts to a neutral nuclear particle, increasing the relative count of neutral particles in the nucleus. What neutral particle is produced? Answer:",
            "In nuclear pair production of neutrons, a high-energy photon can produce a proton and a neutral nuclear particle in the nuclear Coulomb field. What neutral particle is produced? Answer:",
            "When a nucleus captures a thermal neutral particle in a reactor, it may gamma-decay and become a radioactive isotope of the same element. What particle is captured? Answer:",
            "In the fission of plutonium-239 in a fast reactor, fast neutral particles are released that can sustain the chain reaction without moderation. What particles sustain it? Answer:",
            "In muon spin rotation experiments, neutral nuclear particles in the muon-stopping material affect the local field. What particles are relevant? Answer:",
            "In nuclear structure physics, adding one neutral particle to carbon-12 produces the stable isotope carbon-13. What particle is added? Answer:",
            "Radioactive beta-plus emitters convert a nuclear particle into a neutral one; the resulting neutral product remains in the nucleus. What is this neutral product? Answer:",
            "In neutron halo nuclei such as lithium-11, extra neutral particles extend well beyond the nuclear core. What particles form the extended halo? Answer:",
        ],
        f4=[
            "Unlike the proton, which particle in the nucleus carries no electric charge? Answer:",
            "Unlike the electron (charge −1), which nuclear particle has zero electric charge? Answer:",
            "Unlike the charged proton, which nuclear particle has zero net charge? Answer:",
            "While protons carry positive charge and electrons carry negative charge, which nuclear particle is electrically neutral? Answer:",
            "Unlike the proton (charge +1) and electron (charge −1), which nuclear particle has charge 0? Answer:",
            "Unlike the proton, which nuclear particle can be added to a nucleus without changing the element's atomic number? Answer:",
            "While protons give the nucleus its positive charge, which neutral particle contributes to nuclear mass without contributing charge? Answer:",
            "Unlike the charged particles in an atom, which nuclear particle is electrically neutral? Answer:",
            "Unlike protons (which define the element), which nuclear particle creates different isotopes without changing the element? Answer:",
            "Unlike the electron (outside the nucleus and negatively charged), which neutral particle resides inside the nucleus? Answer:",
            "Unlike alpha particles (charge +2) and beta particles (charge −1), which nuclear particle has zero charge? Answer:",
            "Unlike the massless photon (which has no charge and no mass), which neutral particle has mass but zero electric charge? Answer:",
            "Unlike protons that repel each other electromagnetically, which neutral nuclear particle experiences no electromagnetic self-repulsion? Answer:",
            "The nucleus contains two types of nucleons: protons (positive) and which neutral particles? Answer:",
            "While electrons are expelled in beta-minus decay, which neutral nuclear particle is the source particle that converts in that decay? Answer:",
            "Unlike photons (which are massless), which massive neutral particle is found inside atomic nuclei? Answer:",
            "Unlike electrons and protons, which nuclear particle cannot be detected directly by electromagnetic means due to its zero charge? Answer:",
            "Protons and which neutral nuclear particle together are collectively called nucleons? Answer:",
            "Unlike the positron (charge +1), which stable neutral nuclear particle exists in most atomic nuclei? Answer:",
            "Unlike the neutrino (which is very light and rarely interacts), which massive neutral particle is a major constituent of the atomic nucleus? Answer:",
        ],
    ),

    # ── 4. nuclear_stability_provider → neutron (8+18+18+16 = 60) ────────────
    dict(
        fp="nuclear_stability_provider", correct=" neutron",
        pool=[" electron", " proton", " neutron"],
        exp_internal=["proton", "neutron"],
        multi_token=False, exp_type="core",
        f1=[
            "Which neutral nuclear particle helps stabilise the nucleus against proton-proton electromagnetic repulsion? Answer:",
            "Which particle in the nucleus provides stability by mediating the strong nuclear force without adding electromagnetic repulsion? Answer:",
            "Which uncharged nuclear particle is required in increasing numbers to maintain stability as nuclear charge grows? Answer:",
            "Which neutral nucleon allows heavy nuclei to exist without flying apart from proton repulsion? Answer:",
            "Which nuclear particle, when present in the right number relative to protons, prevents radioactive instability? Answer:",
            "Which particle in the nucleus binds protons together via the strong force without adding to electromagnetic repulsion? Answer:",
            "Which electrically neutral nuclear particle prevents the nucleus from disintegrating due to proton-proton repulsion? Answer:",
            "Which uncharged nuclear constituent mediates the strong force to hold protons together in the nucleus? Answer:",
        ],
        f2=[
            "Heavy atomic nuclei require a neutral particle alongside protons to counteract electromagnetic repulsion via the strong force. What is this neutral particle? Answer:",
            "The nuclear valley of stability shows that heavier nuclei require more neutral particles per proton to remain stable. What are these neutral particles? Answer:",
            "As atomic number increases, more neutral nuclear particles are needed relative to protons because proton-proton repulsion grows. What are these neutral particles? Answer:",
            "The belt of stability in nuclear physics describes the ratio of neutral particles to protons needed for a nucleus to be stable. What are these neutral particles? Answer:",
            "In heavy nuclei like uranium, the ratio of neutral nuclear particles to protons is approximately 1.5:1 for stability. What are these neutral particles? Answer:",
            "Nuclear binding curves show that iron-56 is the most stable nucleus; its stability arises from an optimal ratio of protons and neutral particles. What neutral particles? Answer:",
            "Carbon-12's stability comes partly from equal numbers of protons and neutral nuclear particles. What are these neutral particles? Answer:",
            "Lead-208 is doubly magic and highly stable due to specific shell-closure numbers of protons and neutral nuclear particles. What are the neutral particles? Answer:",
            "The shell model of the nucleus predicts magic numbers for both protons and neutral nuclear particles that confer extra stability. What are these neutral particles? Answer:",
            "In nuclear medicine, radioactive isotopes with too few neutral particles relative to protons decay by positron emission. What neutral particles are they deficient in? Answer:",
            "In the Weizsäcker mass formula, the asymmetry energy term penalises nuclei with unequal numbers of neutral particles and protons. What particles appear in this ratio? Answer:",
            "Bismuth-209, the heaviest stable mononuclidic element, achieves stability through its specific ratio of neutral nuclear particles to protons. What neutral particles? Answer:",
            "Radioactive decay chains end at stable lead or bismuth isotopes because these nuclei have the right balance of protons and neutral particles. What neutral particles? Answer:",
            "The liquid drop model of the nucleus includes a Coulomb repulsion term and a symmetry term; the neutral particles reduce the net Coulomb cost. What neutral particles? Answer:",
            "In nuclear engineering, understanding the ratio of neutral particles to protons in reactor fuel is critical for predicting isotope stability. What neutral particles? Answer:",
            "Pairing energy in nuclear structure shows that nuclei with even numbers of neutral particles are more stable than those with odd numbers. What particles are paired? Answer:",
            "Technetium has no stable isotopes because no neutron-to-proton ratio provides stability at its atomic number. What neutral particles are varied to search for stability? Answer:",
            "The SEMF (semi-empirical mass formula) includes an asymmetry term that penalises large differences between the count of neutral particles and protons. What neutral particles? Answer:",
        ],
        f3=[
            "In alpha decay, a nucleus loses two positively charged particles and two neutral particles, reducing nuclear size to gain stability. What neutral particles are lost? Answer:",
            "When a nucleus has too few neutral particles relative to protons, it undergoes positron emission to convert a proton to a neutral particle. What neutral particle is produced? Answer:",
            "In beta-plus decay, a proton converts to a neutral nuclear particle, increasing the neutral-to-proton ratio for stability. What neutral particle is produced? Answer:",
            "Nuclear fission releases several fast neutral particles; when moderated, these can trigger further fissions to sustain the chain. What neutral particles are released? Answer:",
            "In stellar s-process nucleosynthesis, nuclei accumulate neutral particles one at a time, slowly building heavier stable isotopes. What neutral particles are captured? Answer:",
            "The r-process in neutron star mergers rapidly adds many neutral particles to nuclei, far from the stability line. What particles are added? Answer:",
            "When a nucleus captures a neutral particle in a reactor, it may become an unstable isotope of the same element. What particle is absorbed? Answer:",
            "Spontaneous fission of heavy elements releases neutral particles that play a role in nuclear chain reactions. What neutral particles are released? Answer:",
            "During beta-minus decay, a neutral nuclear particle converts to a proton, adjusting the proton-to-neutral ratio toward stability. What particle converts? Answer:",
            "In neutron star formation, protons and electrons combine under extreme pressure to form neutral nuclear particles. What particles form under this pressure? Answer:",
            "In spallation neutron sources, high-energy protons bombard a heavy target to produce beams of neutral particles used for stability studies. What particles are produced? Answer:",
            "Adding one neutral nuclear particle to a stable nucleus can make it radioactive by pushing it off the stability line. What particle is added? Answer:",
            "In nuclear scattering experiments, beams of neutral particles probe nuclear structure without the electromagnetic interference that charged beams would cause. What particles are used? Answer:",
            "In pile reactors, a moderator slows down neutral particles from fission so they can be captured by uranium-235 to sustain fission. What particles are moderated? Answer:",
            "In the production of artificial radioisotopes, stable nuclei capture neutral nuclear particles in a reactor to become unstable. What particles are captured? Answer:",
            "In neutron halo nuclei, extra neutral particles extend well beyond the nuclear core, forming a diffuse halo. What particles form the halo? Answer:",
            "During a nuclear excursion, delayed neutral particles from fission products play a crucial role in reactor safety control. What particles are delayed? Answer:",
            "In muon-catalysed fusion, muons help bring two light nuclei together; the neutral particles inside those nuclei then fuse via the strong force. What neutral particles fuse? Answer:",
        ],
        f4=[
            "Unlike protons (which repel each other via the electromagnetic force), which neutral nuclear particle helps bind protons via the strong force? Answer:",
            "Unlike the proton (which adds both charge and mass to the nucleus), which neutral nuclear particle can be added without changing the element? Answer:",
            "While protons cause electromagnetic repulsion in the nucleus, which neutral particle counteracts this without adding more charge? Answer:",
            "Unlike the electron (which orbits outside the nucleus), which neutral particle is confined inside the nucleus and contributes to stability? Answer:",
            "Unlike adding a proton (which changes the element), adding which neutral nuclear particle creates a new isotope of the same element? Answer:",
            "While protons define the element, which neutral nuclear particles determine the isotope and contribute to nuclear stability? Answer:",
            "Unlike the photon (which is massless), which massive neutral particle inside the nucleus provides stability by participating in the strong force? Answer:",
            "Unlike the alpha particle (positively charged), which neutral nuclear particle can be emitted or captured without changing the element's charge? Answer:",
            "Unlike protons (which cannot be added without changing the element), which neutral particles, when added, only change the isotope? Answer:",
            "While proton-proton repulsion destabilises heavy nuclei, which neutral nuclear particle binds them together via the strong force? Answer:",
            "Unlike pions (mesons that mediate nuclear force), which neutral nucleon actually resides permanently in stable nuclei? Answer:",
            "Unlike the massless neutrino (emitted in beta decay), which massive neutral particle remains in the nucleus providing stability? Answer:",
            "While protons add electromagnetic repulsion to the nucleus, which neutral particle adds binding energy via the strong force without adding repulsion? Answer:",
            "Unlike electrons (which orbit the nucleus and can be transferred in chemical reactions), which neutral particle stays inside the nucleus providing stability? Answer:",
            "Unlike the W boson (which mediates weak interactions), which neutral stable nucleon helps keep the nucleus together? Answer:",
            "While adding a proton would change the element entirely, adding which neutral nuclear particle merely changes the isotope while maintaining stability? Answer:",
        ],
    ),

    # ── 5. em_quantum → photon (10+25+22+18 = 75) ────────────────────────────
    dict(
        fp="em_quantum", correct=" photon",
        pool=[" photon", " electron", " proton"],
        exp_internal=["photon", "electron"],
        multi_token=False, exp_type="core",
        f1=[
            "Which particle is the quantum of electromagnetic radiation? Answer:",
            "Which massless particle carries a discrete packet of electromagnetic energy? Answer:",
            "What is the name of the discrete unit of light energy? Answer:",
            "Which particle constitutes visible light at the quantum level? Answer:",
            "Which massless boson is emitted when an atom undergoes a downward energy transition? Answer:",
            "Which particle carries electromagnetic energy proportional to its frequency? Answer:",
            "What is the quantum of the electromagnetic field called? Answer:",
            "Which fundamental particle is the carrier of both light and electromagnetic radiation in general? Answer:",
            "Which massless, chargeless particle is emitted or absorbed during atomic energy transitions? Answer:",
            "Which particle represents a single quantum of electromagnetic energy? Answer:",
        ],
        f2=[
            "Atomic emission spectra arise when atoms release discrete packets of electromagnetic energy as electrons fall to lower energy levels. What is each packet called? Answer:",
            "Einstein's explanation of the photoelectric effect proposed that light consists of discrete energy packets, each carrying energy hν. What are these packets? Answer:",
            "Solar panels convert electromagnetic energy into electricity; the energy arrives in discrete packets absorbed by the semiconductor. What are these packets? Answer:",
            "Lasers emit coherent light by stimulated emission, where atoms release identical discrete packets of electromagnetic energy. What are these packets? Answer:",
            "In atomic fluorescence, a molecule absorbs a discrete electromagnetic energy packet and re-emits it at a slightly lower frequency. What is this packet? Answer:",
            "In gamma-ray astronomy, detectors count discrete high-energy electromagnetic packets arriving from distant astrophysical sources. What are these packets? Answer:",
            "Planck solved the blackbody radiation problem by assuming energy is emitted in discrete packets proportional to frequency. What are these packets now called? Answer:",
            "Medical X-ray imaging works because a beam of discrete electromagnetic energy packets penetrates soft tissue but is absorbed by bone. What are these packets? Answer:",
            "Photosynthesis captures discrete electromagnetic energy packets from sunlight to drive chemical reactions in chlorophyll. What are these packets? Answer:",
            "When a hydrogen atom transitions from n=2 to n=1, it emits a single discrete packet of electromagnetic energy in the ultraviolet. What is this packet? Answer:",
            "In Compton scattering, an X-ray interacts with an electron by exchanging a discrete electromagnetic energy packet. What is this packet? Answer:",
            "In a photodetector, individual discrete electromagnetic energy packets arrive and trigger an electrical signal. What triggers each signal? Answer:",
            "Raman spectroscopy measures shifts in the energy of discrete electromagnetic packets scattered by molecular vibrations. What are these packets? Answer:",
            "In nuclear gamma decay, the nucleus releases a very high-energy discrete electromagnetic packet. What is this general type of particle? Answer:",
            "In photochemistry, a reaction is triggered when a molecule absorbs a discrete electromagnetic energy packet of the right frequency. What is absorbed? Answer:",
            "In positron emission tomography, two discrete high-energy electromagnetic packets fly in opposite directions after positron annihilation. What are these packets? Answer:",
            "In electron-positron annihilation, two particles convert completely into two high-energy discrete electromagnetic packets. What are the packets? Answer:",
            "In quantum optics, entangled pairs of discrete electromagnetic energy packets are produced for quantum communication. What are these paired particles? Answer:",
            "In radiation therapy, high-energy discrete electromagnetic packets are directed at cancer cells to damage their DNA. What are these packets? Answer:",
            "In fluorescence microscopy, fluorescent labels emit discrete electromagnetic energy packets when excited by light. What are these emitted packets? Answer:",
            "In bioluminescence, chemical reactions in organisms produce discrete electromagnetic energy packets. What are these packets? Answer:",
            "In optical tweezers, focused beams of discrete electromagnetic energy packets trap and manipulate tiny particles. What are the individual packets? Answer:",
            "When a free electron in a plasma decelerates, it emits a discrete electromagnetic energy packet. What is emitted? Answer:",
            "In cavity quantum electrodynamics, a single discrete electromagnetic energy packet is trapped between mirrors and coupled to a single atom. What is trapped? Answer:",
            "In synchrotron radiation, relativistic electrons emit discrete electromagnetic energy packets as they are deflected by magnetic fields. What are these packets? Answer:",
        ],
        f3=[
            "In stimulated emission, an incoming discrete electromagnetic packet triggers an atom to release an identical packet. What is each packet? Answer:",
            "When an excited hydrogen atom falls from the n=3 to n=2 level, it releases a discrete packet of electromagnetic energy in the red visible range. What is this packet? Answer:",
            "In the photoelectric effect, each discrete packet of electromagnetic energy either has enough energy to eject an electron or it does not. What is this packet? Answer:",
            "In pair production, a high-energy discrete electromagnetic packet converts into an electron-positron pair near a nucleus. What is this packet? Answer:",
            "During nuclear de-excitation, the nucleus releases energy as a very high-energy discrete electromagnetic packet. What is this general class of particle? Answer:",
            "In two-photon excitation microscopy, two discrete electromagnetic packets are absorbed simultaneously to excite a fluorophore. What are these packets? Answer:",
            "In spontaneous emission, an excited atom releases a discrete electromagnetic energy packet at a random moment. What is released? Answer:",
            "In absorption spectroscopy, atoms or molecules remove specific discrete electromagnetic packets from a beam of light. What is removed? Answer:",
            "In a photodiode, each discrete electromagnetic packet absorbed in the depletion region creates an electron-hole pair. What is absorbed? Answer:",
            "During synchrotron emission, a relativistic electron in a magnetic field emits discrete electromagnetic packets. What particles are emitted? Answer:",
            "In optical pumping, discrete electromagnetic packets are absorbed to transfer atoms into a specific quantum state. What is absorbed? Answer:",
            "In Cherenkov radiation, a charged particle moving faster than light in a medium produces a cone of discrete electromagnetic packets. What packets are produced? Answer:",
            "In quantum key distribution, single discrete electromagnetic packets carry quantum information that cannot be intercepted without detection. What are these particles? Answer:",
            "In an LED, electrons recombine with holes in a semiconductor junction and emit discrete electromagnetic packets of a specific energy. What is emitted? Answer:",
            "During fluorescence emission, a molecule absorbs a high-energy discrete packet and emits a lower-energy one of the same type. What is emitted? Answer:",
            "In Mössbauer spectroscopy, discrete high-energy electromagnetic packets are emitted and absorbed without nuclear recoil. What are these packets? Answer:",
            "In Bragg diffraction of X-rays by crystals, discrete packets of electromagnetic energy scatter from atomic planes. What are these packets? Answer:",
            "In optical parametric amplification, a strong pump beam of discrete electromagnetic packets amplifies a weaker signal beam. What are the force carriers? Answer:",
            "In a photomultiplier tube, a single incoming discrete electromagnetic packet triggers a cascade of secondary emissions at each dynode. What packet arrives first? Answer:",
            "In Doppler laser cooling, discrete electromagnetic packets carry momentum that slows down atoms in a trap. What are these packets? Answer:",
            "In inverse Compton scattering, a low-energy discrete electromagnetic packet gains energy by scattering off a high-energy electron. What gains energy? Answer:",
            "During Hawking radiation (theoretical), a black hole emits discrete electromagnetic packets and other particles due to quantum vacuum fluctuations. What electromagnetic particles are emitted? Answer:",
        ],
        f4=[
            "Unlike the electron, which massless particle is emitted when an atom undergoes a downward energy transition? Answer:",
            "Unlike the proton (a massive charged particle), which massless uncharged particle carries electromagnetic energy? Answer:",
            "Unlike the neutron (electrically neutral and massive), which massless neutral particle carries electromagnetic radiation? Answer:",
            "Unlike the W and Z bosons (which are massive and mediate the weak force), which massless boson mediates and carries electromagnetic energy? Answer:",
            "Unlike the electron (which has mass and charge), which massless particle mediates atomic energy transitions through emission and absorption? Answer:",
            "Unlike the Higgs boson (which gives mass to particles), which massless boson carries electromagnetic energy in discrete packets? Answer:",
            "While electrons carry charge through wires, which massless particle carries electromagnetic energy through free space? Answer:",
            "Unlike matter particles (which have rest mass), which massless particle always travels at the speed of light and carries electromagnetic energy? Answer:",
            "Unlike the alpha particle (charge +2, massive), which massless, chargeless particle carries electromagnetic energy? Answer:",
            "While the neutrino is nearly massless and carries weak force signals, which massless particle carries electromagnetic energy? Answer:",
            "Unlike the graviton (hypothetical, carries gravity), which confirmed massless particle carries electromagnetic energy? Answer:",
            "Unlike the gluon (which carries the strong force and is confined), which massless boson carries electromagnetic energy freely through space? Answer:",
            "Unlike positrons (which have the same mass as electrons but positive charge), which massless particle has no rest mass and no charge? Answer:",
            "Unlike the massive W+ boson that mediates charged current weak interactions, which massless particle mediates electromagnetic energy transfer? Answer:",
            "While alpha and beta particles are massive, which massless particle is also emitted during nuclear de-excitation (gamma decay)? Answer:",
            "Unlike the phonon (a vibration quantum in a solid), which massless particle carries electromagnetic energy through a vacuum? Answer:",
            "Unlike the pion (massive, mediates residual nuclear force), which massless particle carries electromagnetic energy between charged particles? Answer:",
            "Unlike dark matter candidates (massive hypothetical particles), which confirmed massless particle carries the energy of visible light? Answer:",
        ],
    ),

    # ── 6. massless_em_mediator → photon (6+28+22+21 = 77) ───────────────────
    dict(
        fp="massless_em_mediator", correct=" photon",
        pool=[" photon", " electron", " proton", " neutron"],
        exp_internal=["photon", "electron", "proton"],
        multi_token=False, exp_type="core",
        f1=[
            "Which massless particle mediates the electromagnetic force between charged particles? Answer:",
            "Which force carrier of electromagnetism has zero rest mass? Answer:",
            "Which massless gauge boson is exchanged between charged particles to produce electromagnetic force? Answer:",
            "Which massless boson is the mediator of the electromagnetic interaction in the Standard Model? Answer:",
            "Which particle, when exchanged between two charges, gives rise to the electromagnetic force? Answer:",
            "Which zero-mass particle is the force carrier of electromagnetism? Answer:",
        ],
        f2=[
            "In quantum electrodynamics (QED), the electromagnetic force between two charged particles arises from the exchange of a massless boson. What is this boson? Answer:",
            "The Standard Model identifies four fundamental forces, each with a force-carrying particle. The carrier of electromagnetism is massless. What is it? Answer:",
            "In Feynman diagrams, electromagnetic interactions between charges are shown as an exchange of a massless particle. What particle is exchanged? Answer:",
            "Unlike the W and Z bosons (which give the weak force a short range due to their mass), the electromagnetic force has infinite range because its carrier is massless. What is the carrier? Answer:",
            "In particle physics, virtual particles mediate forces; the virtual particle responsible for electromagnetic repulsion between two electrons is a massless boson. What boson? Answer:",
            "The electromagnetic force has infinite range because its mediating particle is massless and can travel arbitrarily far. What is this mediating particle? Answer:",
            "In QED, the coupling constant (fine structure constant α ≈ 1/137) describes how strongly charged particles interact by exchanging a massless particle. What particle? Answer:",
            "In the electroweak unification, below the symmetry-breaking scale, only the electromagnetic mediator remains massless. What is it? Answer:",
            "Radio antennas emit electromagnetic radiation; at the quantum level, the energy is carried away by massless force-mediating particles. What are they? Answer:",
            "In plasma physics, electromagnetic waves propagate through ionised gas via excitations of the massless electromagnetic field quantum. What are these quanta? Answer:",
            "In Coulomb's law, the force between two charges can be understood quantum mechanically as arising from massless particle exchange. What particle is exchanged? Answer:",
            "The massless nature of the electromagnetic force carrier means electromagnetic fields can extend to astronomical distances. What is this carrier? Answer:",
            "In quantum optics, the coherent state of electromagnetic radiation is described in terms of superpositions of states with different numbers of massless bosons. What bosons? Answer:",
            "Cosmic microwave background radiation fills the universe with massless electromagnetic mediating particles from the Big Bang. What particles fill the universe? Answer:",
            "In electromagnetic shielding (Faraday cage), the arrangement of charges prevents massless electromagnetic force carriers from penetrating. What carriers are blocked? Answer:",
            "In antenna theory, oscillating charges emit massless particles that carry electromagnetic energy to distant receivers. What particles are emitted? Answer:",
            "In classical electrodynamics, Maxwell's equations predict electromagnetic waves; in quantum field theory these waves consist of massless force-carrying particles. What particles? Answer:",
            "In synchrotron light sources, accelerating electrons emit massless electromagnetic force-carrying particles used for scientific experiments. What particles are emitted? Answer:",
            "The photon field in QED is the quantum field whose excitations are the massless mediating particles of electromagnetism. What particles are the excitations? Answer:",
            "In free electron lasers, relativistic electrons emit coherent beams of massless electromagnetic mediating particles. What particles are in the beam? Answer:",
            "In Thomson scattering, a massless electromagnetic particle scatters off a free electron without any energy transfer. What is this massless particle? Answer:",
            "In Rayleigh scattering (which makes the sky blue), massless electromagnetic particles scatter off air molecules proportionally to the fourth power of frequency. What are these particles? Answer:",
            "In solar sails, radiation pressure from massless electromagnetic force carriers pushes the sail forward in space. What particles exert this pressure? Answer:",
            "Photovoltaic cells convert the energy of massless electromagnetic force carriers directly into electrical current in a semiconductor junction. What carriers are absorbed? Answer:",
            "In cavity QED, a single massless electromagnetic force carrier is trapped between two mirrors and coupled to a single atom. What particle is trapped? Answer:",
            "In electromagnetically induced transparency, a control beam of massless force carriers modifies how the medium responds to a probe beam. What are these force carriers? Answer:",
            "In optical lattice clocks, atoms are trapped by the standing wave of massless electromagnetic force carriers at a precision frequency. What carriers form the trap? Answer:",
            "In the Casimir effect, zero-point fluctuations of the massless electromagnetic field produce an attractive force between two uncharged plates. Which field's quanta are responsible? Answer:",
        ],
        f3=[
            "In pair production, a massless electromagnetic mediating particle converts into an electron and a positron near a nucleus. What is this massless particle? Answer:",
            "In Compton scattering, a massless electromagnetic mediating particle collides with an electron and transfers momentum to it. What particle collides? Answer:",
            "When an electron and positron annihilate, they produce two massless electromagnetic force-carrying particles flying in opposite directions. What particles are produced? Answer:",
            "In photoionisation of hydrogen, a massless electromagnetic particle delivers enough energy to remove the bound electron. What particle causes this? Answer:",
            "In the photoelectric effect, each massless electromagnetic force carrier either has sufficient energy to free an electron or it does not. What carrier is involved? Answer:",
            "In spontaneous parametric down-conversion, a massless electromagnetic particle splits into two entangled massless particles of lower energy. What particles are involved? Answer:",
            "During bremsstrahlung, decelerating electrons emit massless electromagnetic force-carrying particles. What particles are emitted? Answer:",
            "In photodisintegration, a high-energy massless electromagnetic mediating particle breaks apart an atomic nucleus. What breaks the nucleus? Answer:",
            "In inverse Compton scattering, a low-energy massless electromagnetic particle gains energy by scattering off a high-energy electron. What gains energy? Answer:",
            "In two-photon physics at colliders, two massless electromagnetic force carriers collide and produce particle pairs. What collide? Answer:",
            "During fluorescence emission, an excited molecule emits a massless electromagnetic force carrier as it returns to the ground state. What is emitted? Answer:",
            "In the photoelectric effect in X-ray photoelectron spectroscopy, massless electromagnetic force carriers eject inner-shell electrons for analysis. What ejects the electrons? Answer:",
            "In optical fibres, information is encoded in pulses of massless electromagnetic force carriers that travel with minimal loss. What carries the information? Answer:",
            "In multiphoton ionisation, an atom absorbs several massless electromagnetic force carriers simultaneously to eject an electron. What are absorbed? Answer:",
            "In solar irradiance, the sun delivers energy to Earth via massless electromagnetic force-carrying particles at a rate of ~1361 W/m². What are these particles? Answer:",
            "In Cherenkov radiation detectors, the massless electromagnetic force carriers emitted by fast charged particles are collected by photomultipliers. What particles are collected? Answer:",
            "In attosecond science, ultrashort bursts of massless electromagnetic force-carrying particles resolve electron dynamics on femtosecond timescales. What bursts are used? Answer:",
            "In optical parametric oscillators, energy is transferred from a pump beam of massless electromagnetic particles to two lower-frequency beams. What particles carry the energy? Answer:",
            "In laser-driven wakefield acceleration, massless electromagnetic force carriers in the laser pulse drive a plasma wave that accelerates electrons. What force carriers drive the wave? Answer:",
            "In gamma-ray bursts, massive quantities of massless electromagnetic force carriers are released in brief intense flashes. What particles are released? Answer:",
            "In Hawking radiation (theoretical), black holes emit massless electromagnetic force carriers among other particles due to quantum vacuum effects. What massless carriers are emitted? Answer:",
            "During stimulated Raman scattering, an incoming massless electromagnetic particle loses energy to a molecular vibration and emerges at lower frequency. What particle is involved? Answer:",
        ],
        f4=[
            "Unlike the W and Z bosons (which mediate the weak force and have significant mass), which massless boson mediates electromagnetism? Answer:",
            "Unlike the gluon (which is also massless but mediates the strong force and is confined), which massless boson mediates electromagnetism freely? Answer:",
            "Unlike the Higgs boson (which has mass and gives other particles their mass), which massless boson mediates electromagnetic interactions? Answer:",
            "Unlike the graviton (hypothetical, massless, but mediates gravity), which confirmed massless particle mediates electromagnetism? Answer:",
            "Unlike massive charged particles like electrons and protons, which massless, uncharged particle mediates the electromagnetic force? Answer:",
            "Unlike the Z boson (which mediates neutral current weak interactions), which massless boson mediates electromagnetic interactions? Answer:",
            "Unlike the W+ boson (charged weak force mediator with significant mass), which massless boson mediates electromagnetism? Answer:",
            "Unlike the pion (which mediates residual strong force between nucleons and has mass), which massless boson mediates electromagnetism? Answer:",
            "The weak force is short-range due to the mass of its force carriers; in contrast, electromagnetism has infinite range because its carrier is massless. What is this massless carrier? Answer:",
            "Unlike the massive carriers of the weak force, which zero-mass carrier transmits the electromagnetic force over infinite distances? Answer:",
            "Unlike the charged W± bosons (which mediate charged current interactions), which neutral massless boson mediates electromagnetism? Answer:",
            "Unlike protons and neutrons (which have rest mass), which massless particle is the quantum of electromagnetic force? Answer:",
            "Unlike the strong force (mediated by gluons) or gravity (mediated by gravitons), the electromagnetic force is mediated by which massless particle? Answer:",
            "Unlike spin-2 gravitons (theoretical, carry gravity), which spin-1 massless particle mediates the electromagnetic force? Answer:",
            "Unlike the inflaton (hypothetical, drove cosmic inflation), which real massless boson mediates electromagnetism today? Answer:",
            "Unlike vector mesons (massive particles in hadronic physics that approximate electromagnetic effects at short range), which massless gauge boson is the true mediator? Answer:",
            "Unlike the massive Z boson that mediates weak neutral currents, which massless boson mediates electromagnetic interactions at all distances? Answer:",
            "Unlike the Kaluza-Klein graviton (hypothetical extra-dimensional massive particle), which confirmed massless particle mediates electromagnetism in four dimensions? Answer:",
            "Unlike the Proca field (a hypothetical massive photon field), the real electromagnetic mediator is massless. What is this real, confirmed mediator? Answer:",
            "Unlike dark matter candidates (hypothetical massive particles that feel gravity but not electromagnetism), which massless boson mediates the electromagnetic force? Answer:",
            "Unlike the muon (a charged, massive lepton), which massless, chargeless boson acts as the force carrier of electromagnetism? Answer:",
        ],
    ),

]  # end CFGS_SINGLE

# ── multi-token generalisation appendix ───────────────────────────────────────

CFGS_MULTI = [

    # ── 7. antielectron → positron (4+8+4+4 = 20) ────────────────────────────
    dict(
        fp="antielectron", correct=" positron",
        pool=[" positron", " electron", " proton", " photon"],
        exp_internal=["positron", "electron", "proton"],
        multi_token=True, exp_type="multi_token_generalisation",
        f1=[
            "Which particle is the antimatter counterpart of the electron? Answer:",
            "Which antiparticle has the same mass as an electron but opposite (positive) charge? Answer:",
            "Which positively charged particle has exactly the same mass as an electron? Answer:",
            "Which particle annihilates with an electron to produce two photons? Answer:",
        ],
        f2=[
            "Dirac's equation predicted a particle identical to the electron but with positive charge. What is this antiparticle? Answer:",
            "When a high-energy photon passes near a nucleus, it can produce a particle-antiparticle pair: an electron and its antimatter partner. What is this antimatter partner? Answer:",
            "PET scans detect pairs of gamma rays produced when electrons annihilate with their antiparticles. What is the antiparticle of the electron? Answer:",
            "Carl Anderson observed a positively charged particle with electron mass in cloud chamber tracks in 1932. What particle had he discovered? Answer:",
            "Dirac's sea model predicted that holes in the negative-energy sea would appear as positively charged particles with electron mass. What are these particles? Answer:",
            "In gamma-ray astronomy, electron-positron annihilation produces pairs of 511 keV gamma rays; the positive particle of this pair is what? Answer:",
            "In accelerator experiments, matter-antimatter pair production creates an electron alongside its positively charged mirror particle. What is this mirror particle? Answer:",
            "In radioactive beta-plus decay, a proton in the nucleus converts to a neutron and emits a positively charged particle with the same mass as an electron. What is this particle? Answer:",
        ],
        f3=[
            "In pair production, a gamma ray creates an electron and its antimatter equivalent near a nucleus. What is this antimatter equivalent? Answer:",
            "In beta-plus radioactive decay, a proton in the nucleus emits an antiparticle of the electron. What is this antiparticle? Answer:",
            "When electron-positron annihilation occurs in PET imaging, an electron and a specific antimatter particle collide. What antimatter particle is involved? Answer:",
            "In inverse pair production (pair annihilation), an electron and its antiparticle collide and produce two photons. What is the electron's antiparticle? Answer:",
        ],
        f4=[
            "Unlike the electron which has negative charge, which antimatter particle has positive charge but the same mass as an electron? Answer:",
            "Unlike the proton (a baryon with charge +1 but much greater mass), which lighter antimatter particle shares the electron's mass but has positive charge? Answer:",
            "Unlike the muon (which is a second-generation lepton with greater mass), which first-generation antimatter particle has the same mass as an electron but opposite charge? Answer:",
            "Unlike the neutron (which has no charge), which antimatter particle has positive charge and is the antiparticle of the electron? Answer:",
        ],
    ),

    # ── 8. heavy_negative_lepton → muon (4+8+4+4 = 20) ──────────────────────
    dict(
        fp="heavy_negative_lepton", correct=" muon",
        pool=[" muon", " electron", " proton", " neutron"],
        exp_internal=["muon", "electron"],
        multi_token=True, exp_type="multi_token_generalisation",
        f1=[
            "Which particle is often called a heavy electron because it has the same charge but much greater mass? Answer:",
            "Which second-generation lepton has the same charge as an electron but is about 207 times heavier? Answer:",
            "Which unstable lepton is produced abundantly in cosmic ray showers in Earth's atmosphere? Answer:",
            "Which negatively charged lepton decays into an electron and two neutrinos with a lifetime of about 2.2 microseconds? Answer:",
        ],
        f2=[
            "In particle physics, the second-generation charged lepton has the same charge as an electron but is approximately 207 times more massive and decays in about 2.2 µs. What is this particle? Answer:",
            "Cosmic rays produce a particle in the upper atmosphere that penetrates deep into the Earth due to its mass and relativistic time dilation. It has the same charge as an electron but far greater mass. What is this particle? Answer:",
            "The lepton family has three generations of charged particles; the second-generation charged lepton is heavier than the electron but lighter than the tau. What is it? Answer:",
            "Time dilation was experimentally confirmed by measuring the survival rate of short-lived, negatively charged leptons produced by cosmic rays at high altitude. What particles were measured? Answer:",
            "Muon spin resonance spectroscopy uses the precession of a negatively charged second-generation lepton in a magnetic field to probe material properties. What lepton is used? Answer:",
            "In the g-2 experiment at Fermilab, the anomalous magnetic moment of a heavy negatively charged lepton is measured with extreme precision. What lepton is studied? Answer:",
            "In muon tomography, naturally occurring second-generation negatively charged leptons from cosmic rays are used to image the interior of volcanoes and pyramids. What particles are used? Answer:",
            "In the Standard Model's lepton sector, there are three charged leptons; the middle one (second generation) is about 207 times heavier than the lightest. What is it? Answer:",
        ],
        f3=[
            "In the decay of a charged pion, a second-generation negatively charged lepton is produced alongside a neutrino. What is this lepton? Answer:",
            "In pion decay chains, the pion decays to a heavy negative lepton, which then itself decays to an electron and two neutrinos. What is the heavy negative lepton? Answer:",
            "In cosmic ray air showers, primary protons interact with atmospheric nuclei to produce pions, which then decay to heavy negatively charged leptons. What are these leptons? Answer:",
            "In neutrino detection experiments, muon neutrinos interact with matter and produce a charged second-generation lepton as evidence of the interaction. What charged lepton is produced? Answer:",
        ],
        f4=[
            "Unlike the electron, which heavier lepton of the same charge is produced in cosmic ray showers and has mass about 207 times greater? Answer:",
            "Unlike the electron (which is stable), which heavier, negatively charged second-generation lepton decays in about 2.2 microseconds? Answer:",
            "Unlike the tau lepton (the heaviest charged lepton), which intermediate-mass negatively charged lepton is the second-generation charged lepton? Answer:",
            "Unlike the electron (stable and lightest charged lepton), which unstable negatively charged lepton is 207 times more massive? Answer:",
        ],
    ),

]  # end CFGS_MULTI

# ── build and write records ───────────────────────────────────────────────────

def dedup(records):
    seen, unique = set(), []
    for r in records:
        key = (r["prompt"], r["filter_property"], r["target_candidate"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

single_records = []
for cfg in CFGS_SINGLE:
    single_records.extend(make_records(cfg))

multi_records = []
for cfg in CFGS_MULTI:
    multi_records.extend(make_records(cfg))

single_unique = dedup(single_records)
multi_unique  = dedup(multi_records)

random.shuffle(single_unique)   # shuffle main corpus
# multi stays in filter order at end so --n_prompts 447 naturally excludes them

output = single_unique + multi_unique

with open(OUT_PATH, "w") as f:
    for r in output:
        f.write(json.dumps(r) + "\n")

# ── summary ───────────────────────────────────────────────────────────────────
from collections import Counter

print(f"Written {len(output)} prompts to {OUT_PATH}")
print(f"  Single-token (main corpus): {len(single_unique)}")
print(f"  Multi-token (appendix):     {len(multi_unique)}")
print()
print("By filter_correct_id:")
for k, v in sorted(Counter(r["filter_correct_id"] for r in output).items()):
    mt = "  [MULTI-TOKEN]" if any(
        r["multi_token_answer"] for r in output if r["filter_correct_id"] == k
    ) else ""
    print(f"  {k}: {v}{mt}")
print()
print("By wording_family (single-token only):")
for k, v in sorted(Counter(r["wording_family"] for r in single_unique).items()):
    print(f"  {k}: {v}")
print()
print("By experiment_type:")
for k, v in sorted(Counter(r["experiment_type"] for r in output).items()):
    print(f"  {k}: {v}")
print()
mt_count = sum(1 for r in output if r["multi_token_answer"])
print(f"Multi-token answer prompts: {mt_count}/{len(output)} = {mt_count/len(output):.1%}")
print()
pools = Counter(tuple(r["implicit_candidate_pool"]) for r in output)
print("Implicit candidate pools:")
for pool, count in sorted(pools.items(), key=lambda x: -x[1]):
    print(f"  {list(pool)}: {count}")
print()
print(f"Correct answer distribution (single-token):")
for k, v in sorted(Counter(r["target_candidate"] for r in single_unique).items()):
    print(f"  {k}: {v} ({v/len(single_unique):.1%})")
print()
lens = [len(r["prompt"].split()) for r in single_unique]
print(f"Single-token prompt word length: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}")
