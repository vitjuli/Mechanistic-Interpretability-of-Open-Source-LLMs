"""
Generate prompts for physics_entropy_gate_v1.

Scientific hypothesis: around L28, Qwen3-4B forms a thermodynamic gate-state
representation that separates physically allowed from physically blocked processes.
This behaviour tests the second-law entropy gate across wording families, difficulty
levels, and adversarial near-misses.

Design:
  20 minimal pairs (allow/block) × 8 wording families = 320 prompts.
  Adversarial pairs use only W0/W1/W2/W5 (4 wordings) to avoid ambiguity.
  Final count depends on per-concept wording_mask; target ~300-320 prompts.

Answer tokens (Qwen3-4B tokenizer, confirmed single tokens in script 80/81):
  ' Yes' (token 7414) — process is physically possible/allowed
  ' No'  (token 2308) — process is physically impossible/blocked

Minimal-pair structure:
  Each concept has:
    pair_id       : e.g. "C01"
    pair_role     : "allow" | "block"
    gate_label    : "allow" | "block"
    correct_answer: " Yes" (allow) | " No" (block)

Output:
  data/prompts/physics_entropy_gate_v1_train.jsonl

Usage:
    python scripts/85_generate_entropy_gate_prompts.py [--seed 42] [--out PATH]
"""

import argparse
import json
import random
from pathlib import Path

BEHAVIOUR = "physics_entropy_gate_v1"
DEFAULT_OUT = Path("data/prompts/physics_entropy_gate_v1_train.jsonl")

YES_STR = " Yes"
NO_STR  = " No"


# ── Wording families ──────────────────────────────────────────────────────────
# Each template receives:
#   {core}  = the physical scenario description (lowercase, no trailing period)
#   {Core}  = sentence-case version
#   {sym}   = symbolic form (ΔS, Q, etc.) — only for W4 capable concepts

WORDING_TEMPLATES = {
    "W0_direct": "Is it physically possible for {core}? Answer:",
    "W1_spontaneous": "Can {core} happen spontaneously, without any external work or energy input? Answer:",
    "W2_second_law": "Is {core} consistent with the second law of thermodynamics? Answer:",
    "W3_questioning": "Does the second law of thermodynamics permit {core}? Answer:",
    "W4_symbolic": None,   # per-concept symbolic template — set in CONCEPTS
    "W5_adversarial": None,  # per-concept adversarial framing — set in CONCEPTS
    "W6_minimal_pair": None,  # short contrast framing — set in CONCEPTS
    "W7_experimental": None,  # lab context — set in CONCEPTS
}

ALL_WORDING_KEYS = list(WORDING_TEMPLATES.keys())


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# ── Concept definitions ───────────────────────────────────────────────────────
# Each entry is a minimal pair dict with:
#   pair_id, concept, domain, constraint_type, abstract_rule, difficulty
#   allowed_core, blocked_core
#   physical_direction_allow, physical_direction_block
#   wording_mask: list of wording family keys to generate (all 8 or subset)
#   W4_allow, W4_block: symbolic templates (None if no symbolic form)
#   W5_allow, W5_block: adversarial framing templates
#   W6_allow, W6_block: minimal-pair short framing
#   W7_allow, W7_block: experimental-context framing

CONCEPTS = [

    # ── EASY: Named physical processes ────────────────────────────────────────

    dict(
        pair_id="C01", concept="heat_flow",
        domain="thermodynamics", constraint_type="second_law_clausius",
        abstract_rule="heat_flows_spontaneously_hot_to_cold",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="heat to flow spontaneously from a hot body to a colder body",
        blocked_core="heat to flow spontaneously from a cold body to a hotter body without any external work",
        physical_direction_allow="hot_to_cold",
        physical_direction_block="cold_to_hot_no_work",
        W4_allow="ΔS_universe > 0 when heat flows from the hot to the cold reservoir",
        W4_block="ΔS_universe < 0 when heat flows spontaneously from the cold to the hot reservoir",
        W5_allow="Heat naturally moves from warmer to cooler objects, and this process requires no energy input. Is that correct? Answer:",
        W5_block="Because a refrigerator moves heat from cold to hot, can heat always spontaneously flow from cold to hot on its own? Answer:",
        W6_allow="Hot body transfers heat to cold body — allowed? Answer:",
        W6_block="Cold body spontaneously heats a hotter body — allowed? Answer:",
        W7_allow="A metal bar at 80°C is placed in contact with a metal bar at 20°C. Is it physically possible for heat to flow from the hot bar to the cold bar? Answer:",
        W7_block="A metal bar at 20°C is placed in contact with a metal bar at 80°C. Is it physically possible for heat to flow spontaneously from the cold bar to the hot bar, with no refrigerator or other energy source present? Answer:",
    ),

    dict(
        pair_id="C02", concept="entropy_isolated_system",
        domain="thermodynamics", constraint_type="second_law_entropy",
        abstract_rule="entropy_cannot_spontaneously_decrease_in_isolated_system",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="the total entropy of an isolated system to increase during a spontaneous process",
        blocked_core="the total entropy of an isolated system to decrease spontaneously, with no external interaction",
        physical_direction_allow="entropy_increase",
        physical_direction_block="entropy_decrease",
        W4_allow="ΔS_total ≥ 0 for any spontaneous process in an isolated system",
        W4_block="ΔS_total < 0 to hold for a spontaneous process in a perfectly isolated system",
        W5_allow="Since entropy always increases in isolated systems, it must be possible for entropy to increase spontaneously. Is that consistent with physics? Answer:",
        W5_block="If a system is perfectly isolated, could local fluctuations cause its overall entropy to decrease permanently over time? Answer:",
        W6_allow="Isolated system: entropy increases spontaneously — allowed? Answer:",
        W6_block="Isolated system: entropy decreases spontaneously — allowed? Answer:",
        W7_allow="Consider a sealed, thermally insulated container with a gas expanding into a vacuum. Is it possible for the total entropy of the system to increase? Answer:",
        W7_block="Consider a sealed, thermally insulated container with a gas at equilibrium. Is it physically possible for the gas to spontaneously concentrate itself into one half of the container, decreasing its total entropy, with no external input? Answer:",
    ),

    dict(
        pair_id="C03", concept="thermal_equilibration",
        domain="thermodynamics", constraint_type="zeroth_second_law",
        abstract_rule="objects_in_contact_reach_common_equilibrium_temperature",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="two objects at different temperatures, placed in thermal contact, to gradually reach the same equilibrium temperature",
        blocked_core="the colder of two objects in thermal contact to spontaneously become colder while the hotter object becomes hotter, without any external energy",
        physical_direction_allow="thermal_equilibration",
        physical_direction_block="anti_equilibration",
        W4_allow=None,
        W4_block=None,
        W5_allow="Since heat flows from hot to cold, objects in contact must eventually reach the same temperature. Is that physically allowed? Answer:",
        W5_block="In a closed system, could a temperature gradient spontaneously grow larger, making hot objects hotter and cold objects colder, with no external driving force? Answer:",
        W6_allow="Two bodies in contact → reach common temperature over time — allowed? Answer:",
        W6_block="Two bodies in contact → temperature difference spontaneously grows — allowed? Answer:",
        W7_allow="A warm cup of coffee is left in a cool room. Is it physically possible for the coffee to eventually reach room temperature? Answer:",
        W7_block="A warm cup of coffee is left in a cool room. Is it physically possible for the coffee to become hotter while the room becomes cooler, with no energy source? Answer:",
    ),

    dict(
        pair_id="C04", concept="refrigerator_external_work",
        domain="thermodynamics", constraint_type="second_law_kelvin_clausius",
        abstract_rule="heat_pump_cold_to_hot_requires_external_work",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="a heat pump or refrigerator to transfer heat from a cold reservoir to a hot reservoir when external work is continuously supplied",
        blocked_core="heat to transfer continuously from a cold reservoir to a hot reservoir with no external work input and no other change in the surroundings",
        physical_direction_allow="cold_to_hot_with_work",
        physical_direction_block="cold_to_hot_no_work",
        W4_allow="W_input > 0 allows Q_cold→hot > 0 in a refrigeration cycle, consistent with ΔS_universe ≥ 0",
        W4_block="ΔS_universe < 0 for spontaneous cold-to-hot heat transfer with W = 0 — this violates the second law",
        W5_allow="A refrigerator moves heat from cold food to the warm room. Since refrigerators work, is it physically possible for heat to move from cold to hot when work is applied? Answer:",
        W5_block="Since a refrigerator moves heat from cold to hot, can this process happen spontaneously without plugging the refrigerator in? Answer:",
        W6_allow="Heat pump with external work: cold to hot — allowed? Answer:",
        W6_block="Heat pump with no energy input: cold to hot — allowed? Answer:",
        W7_allow="An electric refrigerator is powered by an electrical outlet. Is it physically possible for this refrigerator to move heat from the cold interior to the warm kitchen? Answer:",
        W7_block="A refrigerator is unplugged. Is it physically possible for the refrigerator to continue transferring heat from its cold interior to the warm kitchen, with no electrical power or other energy source? Answer:",
    ),

    dict(
        pair_id="C05", concept="free_expansion",
        domain="thermodynamics", constraint_type="second_law_entropy",
        abstract_rule="gas_expands_freely_into_vacuum_entropy_increases",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="an ideal gas to expand spontaneously and irreversibly into an evacuated chamber when a partition is removed",
        blocked_core="an ideal gas that has expanded into an evacuated chamber to spontaneously re-compress itself back into its original half of the container, with no pistons or external forces",
        physical_direction_allow="free_expansion",
        physical_direction_block="spontaneous_compression",
        W4_allow=None,
        W4_block=None,
        W5_allow="When a valve is opened between a gas-filled flask and an evacuated flask, the gas rushes in to fill both. Is this physically allowed? Answer:",
        W5_block="After a gas has filled both flasks, could the molecules spontaneously rush back to fill only one flask, leaving the other in vacuum? Answer:",
        W6_allow="Gas expands into vacuum when partition removed — allowed? Answer:",
        W6_block="Expanded gas spontaneously recompresses into half the volume — allowed? Answer:",
        W7_allow="A sealed container is divided by a partition. One side has ideal gas at pressure P; the other is evacuated. The partition is removed. Is it physically possible for the gas to expand to fill the entire container? Answer:",
        W7_block="After the gas has expanded to fill the entire container, is it physically possible for the gas molecules to spontaneously concentrate themselves back into just the original half, leaving a perfect vacuum in the other half, with no outside intervention? Answer:",
    ),

    dict(
        pair_id="C06", concept="ice_melting_direction",
        domain="thermodynamics", constraint_type="second_law_phase_transition",
        abstract_rule="phase_transition_direction_governed_by_entropy_and_temperature",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="ice at atmospheric pressure to melt spontaneously when placed in a room at 25°C",
        blocked_core="liquid water at atmospheric pressure to spontaneously freeze into solid ice in a room at 25°C, without any cooling device or heat removal",
        physical_direction_allow="melting_above_melting_point",
        physical_direction_block="spontaneous_freezing_at_room_temp",
        W4_allow="ΔG_melting < 0 at T = 25°C for water at atmospheric pressure, so melting is spontaneous",
        W4_block="ΔG_freezing > 0 at T = 25°C for water at atmospheric pressure, so spontaneous freezing is prohibited",
        W5_allow="Water freezes at 0°C, so at 25°C (above the melting point) ice must melt. Is this physically allowed? Answer:",
        W5_block="Water can exist as a solid below 0°C. Could a glass of liquid water at 25°C spontaneously turn into ice without a freezer or any cooling? Answer:",
        W6_allow="Ice at 25°C melts spontaneously — allowed? Answer:",
        W6_block="Water at 25°C spontaneously freezes — allowed? Answer:",
        W7_allow="A tray of ice cubes is taken out of the freezer and left on a counter at 25°C. Is it physically possible for the ice to melt? Answer:",
        W7_block="A glass of liquid water is left on a counter at 25°C. Is it physically possible for this water to spontaneously turn into ice, without any refrigeration or cooling device? Answer:",
    ),

    # ── MEDIUM: Rule-application required ────────────────────────────────────

    dict(
        pair_id="C07", concept="carnot_efficiency_limit",
        domain="thermodynamics", constraint_type="second_law_carnot",
        abstract_rule="real_engine_efficiency_cannot_exceed_carnot_efficiency",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="a real heat engine operating between reservoirs at 300 K and 600 K to achieve a thermal efficiency of 40%",
        blocked_core="a real heat engine operating between reservoirs at 300 K and 600 K to achieve a thermal efficiency of 60%, which exceeds the Carnot limit of 50%",
        physical_direction_allow="below_carnot_limit",
        physical_direction_block="above_carnot_limit",
        W4_allow="η = 0.40 < η_Carnot = 1 − T_cold/T_hot = 1 − 300/600 = 0.50, so this is consistent with the second law",
        W4_block="η = 0.60 > η_Carnot = 0.50 for T_cold = 300 K, T_hot = 600 K — this violates the second law",
        W5_allow="The Carnot efficiency between 300 K and 600 K is 50%. A real engine achieving 40% is below this limit. Is that physically possible? Answer:",
        W5_block="The Carnot efficiency sets an upper bound. Could an engineer build a heat engine between 300 K and 600 K that is 60% efficient, beating the Carnot limit? Answer:",
        W6_allow="Real engine at 40% between 300 K and 600 K (Carnot = 50%) — allowed? Answer:",
        W6_block="Real engine at 60% between 300 K and 600 K (Carnot = 50%) — allowed? Answer:",
        W7_allow="An engineer builds a heat engine that operates between a boiler at 600 K and a condenser at 300 K, achieving 40% thermal efficiency. Is this physically possible? Answer:",
        W7_block="An engineer claims to have built a heat engine operating between a boiler at 600 K and a condenser at 300 K, achieving 60% thermal efficiency. Is this physically possible, given the Carnot limit of 50%? Answer:",
    ),

    dict(
        pair_id="C08", concept="diffusion_direction",
        domain="thermodynamics", constraint_type="second_law_entropy_mixing",
        abstract_rule="diffusion_proceeds_from_high_to_low_concentration",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="dissolved molecules to diffuse spontaneously from a region of higher concentration to a region of lower concentration",
        blocked_core="dissolved molecules to diffuse spontaneously from a region of lower concentration to a region of higher concentration, with no external energy input or semi-permeable membrane",
        physical_direction_allow="high_to_low_concentration",
        physical_direction_block="low_to_high_concentration",
        W4_allow=None,
        W4_block=None,
        W5_allow="Diffusion naturally moves molecules down concentration gradients, from high to low concentration. Is that physically allowed? Answer:",
        W5_block="Since osmosis can concentrate molecules on one side of a membrane, can molecules spontaneously diffuse from lower to higher concentration in a homogeneous solution, with no membrane or energy? Answer:",
        W6_allow="Molecules diffuse from high to low concentration — allowed? Answer:",
        W6_block="Molecules diffuse from low to high concentration without external energy — allowed? Answer:",
        W7_allow="A drop of dye is placed in a beaker of still water. Is it physically possible for the dye to gradually spread out, diffusing from the concentrated drop to the surrounding water? Answer:",
        W7_block="After dye has spread uniformly through a beaker of water, is it physically possible for the dye to spontaneously re-concentrate itself into a small drop, with no external stirring or energy input? Answer:",
    ),

    dict(
        pair_id="C09", concept="entropy_universe_sign",
        domain="thermodynamics", constraint_type="second_law_entropy",
        abstract_rule="ΔS_universe_geq_0_for_any_process",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="the total entropy of the universe to increase or stay the same during any real physical process",
        blocked_core="the total entropy of the universe to decrease during a physical process, with no compensating external mechanism",
        physical_direction_allow="delta_S_universe_nonnegative",
        physical_direction_block="delta_S_universe_negative",
        W4_allow="ΔS_universe ≥ 0 for all physical processes — this is the mathematical statement of the second law",
        W4_block="ΔS_universe < 0 for an actual physical process — this directly violates the second law",
        W5_allow="The second law tells us entropy of the universe can only increase or stay constant. Does this mean ΔS_universe ≥ 0 is physically allowed? Answer:",
        W5_block="Because entropy can decrease locally (e.g. inside a refrigerator), could ΔS_universe be negative for some clever process? Answer:",
        W6_allow="ΔS_universe ≥ 0 for a physical process — allowed? Answer:",
        W6_block="ΔS_universe < 0 for a physical process — allowed? Answer:",
        W7_allow="A physicist computes the total entropy change of the universe during a chemical reaction and finds ΔS_universe = +5 J/K. Is this outcome physically possible? Answer:",
        W7_block="A physicist computes the total entropy change of the universe during a physical process and finds ΔS_universe = −3 J/K. Is this outcome physically possible, with no external system compensating? Answer:",
    ),

    dict(
        pair_id="C10", concept="gas_mixing_spontaneous",
        domain="thermodynamics", constraint_type="second_law_entropy_mixing",
        abstract_rule="mixing_increases_entropy_and_is_spontaneous",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="two different ideal gases to mix spontaneously when the partition separating them is removed",
        blocked_core="a uniform mixture of two ideal gases in a sealed container to spontaneously separate into its two pure components, with no external work or selective membrane",
        physical_direction_allow="mixing_spontaneous",
        physical_direction_block="unmixing_spontaneous",
        W4_allow=None,
        W4_block=None,
        W5_allow="The entropy of mixing is always positive, so mixing is spontaneous. Is it physically allowed for two gases to mix when separated by a partition that is removed? Answer:",
        W5_block="Since a semi-permeable membrane can separate gases, could a mixture of gases spontaneously separate into pure components in a sealed container with no membrane or energy? Answer:",
        W6_allow="Two gases mix spontaneously when partition removed — allowed? Answer:",
        W6_block="Gas mixture spontaneously separates into pure components — allowed? Answer:",
        W7_allow="A sealed box is divided into two compartments — one with nitrogen and one with oxygen. The partition is removed. Is it physically possible for the gases to mix? Answer:",
        W7_block="After nitrogen and oxygen have mixed uniformly in a sealed box, is it physically possible for them to spontaneously separate back into pure nitrogen and pure oxygen on opposite sides, with no external intervention? Answer:",
    ),

    dict(
        pair_id="C11", concept="kelvin_planck_statement",
        domain="thermodynamics", constraint_type="second_law_kelvin_planck",
        abstract_rule="cannot_convert_all_heat_to_work_in_cyclic_process",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=ALL_WORDING_KEYS,
        allowed_core="a heat engine operating in a cycle to absorb heat from a hot reservoir, convert part of it into useful work, and reject the remainder to a cold reservoir",
        blocked_core="a device operating in a complete cycle to absorb heat from a single thermal reservoir and convert ALL of it into work, with no other effect whatsoever",
        physical_direction_allow="partial_heat_to_work_with_cold_sink",
        physical_direction_block="complete_heat_to_work_no_cold_sink",
        W4_allow="A cyclic heat engine satisfying η < η_Carnot, with Q_rejected > 0 to the cold reservoir, is consistent with the Kelvin-Planck statement",
        W4_block="W_out = Q_in and Q_rejected = 0 in a cycle: this is a Kelvin-Planck perpetual-motion machine of the second kind",
        W5_allow="Steam engines convert heat into work, but always dump some heat to the condenser. Is this partial conversion physically allowed? Answer:",
        W5_block="Could a perfectly efficient engine absorb heat from the ocean and convert all of it into ship propulsion, with no waste heat at all, running indefinitely? Answer:",
        W6_allow="Cyclic engine converts part of heat to work, rejects rest — allowed? Answer:",
        W6_block="Cyclic engine converts 100% of heat to work, no cold sink — allowed? Answer:",
        W7_allow="An inventor proposes a steam turbine that absorbs 1000 J from a boiler, delivers 350 J of mechanical work, and exhausts 650 J of waste heat to a condenser. Is this physically possible? Answer:",
        W7_block="An inventor proposes an engine that absorbs 1000 J from the ocean (a single reservoir at uniform temperature) and delivers 1000 J of mechanical work per cycle, with absolutely no waste heat. Is this physically possible? Answer:",
    ),

    dict(
        pair_id="C12", concept="friction_heat_direction",
        domain="thermodynamics", constraint_type="second_law_irreversibility",
        abstract_rule="friction_converts_work_to_heat_irreversibly",
        difficulty="easy", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="kinetic energy to be irreversibly converted into thermal energy (heat) when friction acts between two sliding surfaces",
        blocked_core="heat stored in a warm floor to spontaneously convert itself into directed kinetic energy that causes a stationary block to begin sliding, with no external force",
        physical_direction_allow="work_to_heat_by_friction",
        physical_direction_block="heat_to_ordered_kinetic_energy",
        W4_allow=None,
        W4_block=None,
        W5_allow="Rubbing hands together produces heat — friction converts motion into thermal energy. Is that physically allowed? Answer:",
        W5_block="A warm floor contains thermal energy from friction. Could that stored heat spontaneously cause a block resting on the floor to start sliding, converting heat back into ordered motion? Answer:",
        W6_allow="Sliding block loses kinetic energy → heats the floor — allowed? Answer:",
        W6_block="Warm floor spontaneously causes resting block to start moving — allowed? Answer:",
        W7_allow="A wooden block slides across a rough floor and comes to rest. The floor and block are slightly warmer than before. Is it physically possible for kinetic energy to have been converted into heat by friction? Answer:",
        W7_block="A wooden block rests on a warm floor. Is it physically possible for the thermal energy of the floor to spontaneously convert into ordered kinetic energy that causes the block to begin moving, without any external push? Answer:",
    ),

    # ── HARD: Abstracted or symbolic principles ───────────────────────────────

    dict(
        pair_id="C13", concept="reversible_process_limit",
        domain="thermodynamics", constraint_type="second_law_reversibility",
        abstract_rule="reversible_process_has_ΔS_universe_equals_zero",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W3_questioning", "W4_symbolic",
                      "W5_adversarial", "W6_minimal_pair"],
        allowed_core="ΔS_universe = 0 for a perfectly reversible process (the theoretical limit of maximum efficiency)",
        blocked_core="ΔS_universe < 0 for a perfectly reversible process",
        physical_direction_allow="reversible_process_ΔS=0",
        physical_direction_block="reversible_process_ΔS<0",
        W4_allow="ΔS_universe = 0 for a reversible process — this is the theoretical minimum entropy production, achievable only in the ideal reversible limit",
        W4_block="ΔS_universe < 0 for a reversible process — but the second law states ΔS_universe ≥ 0 even for reversible processes (equality holds at the limit)",
        W5_allow="Reversible processes are idealizations where entropy production is zero. Is ΔS_universe = 0 physically allowed for a reversible process? Answer:",
        W5_block="Since reversible processes are more efficient than irreversible ones, could a perfectly reversible process actually achieve ΔS_universe < 0? Answer:",
        W6_allow="Reversible process: ΔS_universe = 0 — allowed? Answer:",
        W6_block="Reversible process: ΔS_universe < 0 — allowed? Answer:",
        W7_allow=None,
        W7_block=None,
    ),

    dict(
        pair_id="C14", concept="entropy_symbolic_expansion",
        domain="thermodynamics", constraint_type="second_law_entropy",
        abstract_rule="ΔS_positive_for_irreversible_expansion",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W4_symbolic", "W5_adversarial",
                      "W6_minimal_pair", "W7_experimental"],
        allowed_core="the entropy change of an ideal gas to be positive (ΔS > 0) when it expands irreversibly into a larger volume at constant temperature",
        blocked_core="the entropy change of an ideal gas to be negative (ΔS < 0) when it expands irreversibly into a larger volume at constant temperature, with no heat removal",
        physical_direction_allow="entropy_positive_on_expansion",
        physical_direction_block="entropy_negative_on_expansion",
        W4_allow="ΔS = nR ln(V₂/V₁) > 0 for V₂ > V₁ — entropy increases on irreversible isothermal expansion",
        W4_block="ΔS = nR ln(V₂/V₁) < 0 is impossible for V₂ > V₁ since ln(V₂/V₁) > 0",
        W5_allow="When a gas expands into more volume, the molecules have more accessible microstates, so entropy increases. Is ΔS > 0 for free expansion correct? Answer:",
        W5_block="Since entropy is a state function and expansion is often accompanied by cooling in real gases, could the entropy change be negative during irreversible expansion into vacuum? Answer:",
        W6_allow="Ideal gas expands isothermally: ΔS > 0 — allowed? Answer:",
        W6_block="Ideal gas expands isothermally: ΔS < 0 — allowed? Answer:",
        W7_allow="A mole of ideal gas expands irreversibly from 1 L to 2 L at constant temperature. Is it physically possible for its entropy to increase? Answer:",
        W7_block="A mole of ideal gas expands irreversibly from 1 L to 2 L at constant temperature, with no heat removed. Is it physically possible for its entropy to decrease? Answer:",
    ),

    # ── ADVERSARIAL: Misleading surface similarity ────────────────────────────

    dict(
        pair_id="C15", concept="local_entropy_decrease_open_system",
        domain="thermodynamics", constraint_type="second_law_open_system",
        abstract_rule="local_entropy_can_decrease_in_open_system_if_ΔS_universe_nonneg",
        difficulty="hard", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W5_adversarial",
                      "W6_minimal_pair", "W7_experimental"],
        allowed_core="the entropy of a local subsystem (such as a growing crystal or a living cell) to decrease, provided the entropy exported to the surroundings is at least as large, so that ΔS_universe ≥ 0",
        blocked_core="the total entropy of the universe to decrease when a local system's entropy decreases, with no compensating entropy increase in the surroundings",
        physical_direction_allow="local_entropy_decrease_with_compensation",
        physical_direction_block="universe_entropy_decrease",
        W4_allow=None,
        W4_block=None,
        W5_allow="Living organisms decrease their local entropy while growing. Is it physically allowed for a local system's entropy to decrease if it exports enough entropy to the surroundings? Answer:",
        W5_block="Since living organisms locally decrease entropy, does this mean the second law permits the total entropy of the universe to decrease overall? Answer:",
        W6_allow="Local system entropy decreases, surroundings entropy increases more — allowed? Answer:",
        W6_block="Total entropy of universe decreases when local system decreases — allowed? Answer:",
        W7_allow="A refrigerator removes heat from its cold interior, decreasing the interior's entropy. Is it physically possible for the local entropy of the refrigerator interior to decrease? Answer:",
        W7_block="A refrigerator decreases the entropy of its cold interior. Does this mean the total entropy of the universe (refrigerator + kitchen + power grid) also decreases? Answer:",
    ),

    dict(
        pair_id="C16", concept="adiabatic_compression_temperature",
        domain="thermodynamics", constraint_type="first_second_law_combined",
        abstract_rule="adiabatic_compression_raises_temperature_irreversibly",
        difficulty="medium", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W1_spontaneous", "W2_second_law", "W3_questioning",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="the temperature of an ideal gas to rise when it is rapidly and irreversibly compressed adiabatically (no heat exchange with surroundings)",
        blocked_core="the temperature of an ideal gas to fall during rapid irreversible adiabatic compression, with no heat exchanged with the surroundings",
        physical_direction_allow="adiabatic_compression_heats",
        physical_direction_block="adiabatic_compression_cools",
        W4_allow=None,
        W4_block=None,
        W5_allow="When a bicycle pump is used, the air gets hot because adiabatic compression raises temperature. Is that physically allowed? Answer:",
        W5_block="Since adiabatic expansion cools a gas, could adiabatic compression also cool a gas, by analogy? Answer:",
        W6_allow="Adiabatic compression: gas temperature rises — allowed? Answer:",
        W6_block="Adiabatic compression: gas temperature falls — allowed? Answer:",
        W7_allow="A piston rapidly compresses a gas in a thermally insulated cylinder. Is it physically possible for the gas temperature to increase? Answer:",
        W7_block="A piston rapidly compresses a gas in a thermally insulated cylinder, doing work on the gas. Is it physically possible for the gas temperature to decrease during this compression, with no heat exchanged? Answer:",
    ),

    dict(
        pair_id="C17", concept="heat_engine_cold_reservoir_required",
        domain="thermodynamics", constraint_type="second_law_kelvin_planck",
        abstract_rule="heat_engine_requires_cold_reservoir_to_reject_heat",
        difficulty="hard", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W3_questioning",
                      "W4_symbolic", "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="a heat engine to require a cold reservoir to dump waste heat into, even though some energy is wasted",
        blocked_core="a heat engine to operate continuously without any cold reservoir to reject waste heat, absorbing heat from a hot source and producing work with zero waste heat",
        physical_direction_allow="engine_with_cold_sink",
        physical_direction_block="engine_without_cold_sink",
        W4_allow="η < 1 requires Q_rejected = Q_in × (1 − η) > 0 to a cold sink, consistent with the Kelvin-Planck statement",
        W4_block="η = 1 with Q_rejected = 0 violates the Kelvin-Planck formulation of the second law",
        W5_allow="All working heat engines, from steam turbines to car engines, must exhaust waste heat to a cold reservoir. Is it physically required for a heat engine to have a cold sink? Answer:",
        W5_block="Modern heat engines are becoming very efficient. Could a sufficiently well-engineered engine eventually operate with no cold reservoir and no waste heat? Answer:",
        W6_allow="Heat engine with cold reservoir (some waste heat) — allowed? Answer:",
        W6_block="Heat engine with no cold reservoir (zero waste heat) — allowed? Answer:",
        W7_allow="A power plant absorbs 1000 MW from burning fuel and generates 400 MW of electricity, rejecting 600 MW as waste heat to a river. Is it physically possible that this engine requires the cold river as a heat sink? Answer:",
        W7_block="An inventor proposes a power plant that absorbs heat from the sun and converts it entirely into electricity, with no cooling tower, no condenser, and no waste heat at all. Is this physically possible? Answer:",
    ),

    dict(
        pair_id="C18", concept="perpetual_motion_second_kind",
        domain="thermodynamics", constraint_type="second_law_kelvin_planck",
        abstract_rule="perpetual_motion_machine_of_second_kind_is_impossible",
        difficulty="hard", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W4_symbolic",
                      "W5_adversarial", "W6_minimal_pair", "W7_experimental"],
        allowed_core="a device to extract mechanical work from a reservoir by also exchanging heat with a second reservoir at a different temperature (a standard heat engine)",
        blocked_core="a perpetual-motion machine of the second kind — a device that extracts heat from a single thermal reservoir at uniform temperature and converts it entirely into work with no other effect, running indefinitely",
        physical_direction_allow="standard_heat_engine_two_reservoirs",
        physical_direction_block="perpetual_motion_second_kind",
        W4_allow="A standard heat engine satisfies ΔS_universe ≥ 0 by using two reservoirs at different temperatures",
        W4_block="A PMM2 would require ΔS_universe < 0 per cycle, violating the second law",
        W5_allow="Steam engines use a temperature difference between a boiler and condenser to produce work. Is a heat engine using two reservoirs at different temperatures physically allowed? Answer:",
        W5_block="The ocean is an enormous thermal reservoir. Could a ship extract heat from the ocean and use it entirely as propulsion, with no cooler reservoir, running indefinitely? Answer:",
        W6_allow="Engine using two reservoirs at different temperatures to produce work — allowed? Answer:",
        W6_block="Engine extracting heat from single uniform-temperature reservoir, all converted to work — allowed? Answer:",
        W7_allow="A submarine uses a thermoelectric generator powered by the temperature difference between warm surface water and cold deep water to produce electricity. Is this physically allowed? Answer:",
        W7_block="A submarine is proposed to extract heat from the ocean at uniform temperature and convert it entirely into electricity for propulsion, running indefinitely with no other energy source. Is this physically possible? Answer:",
    ),

    dict(
        pair_id="C19", concept="clausius_inequality",
        domain="thermodynamics", constraint_type="second_law_clausius_inequality",
        abstract_rule="ΔS_system_geq_Q_over_T_for_irreversible_processes",
        difficulty="hard", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W4_symbolic",
                      "W5_adversarial", "W6_minimal_pair"],
        allowed_core="a system's entropy change to be greater than Q/T for an irreversible process, where Q is the heat absorbed and T is the temperature of the surroundings",
        blocked_core="a system's entropy change to be less than Q/T for an irreversible process, with no other system compensating",
        physical_direction_allow="ΔS_geq_Q_over_T",
        physical_direction_block="ΔS_less_than_Q_over_T",
        W4_allow="Clausius inequality: ΔS ≥ Q/T for any process (equality for reversible), so ΔS > Q/T for irreversible processes is consistent",
        W4_block="ΔS < Q/T for an irreversible process violates the Clausius inequality and is prohibited by the second law",
        W5_allow="The Clausius inequality tells us that the entropy change is always at least Q/T. Is it physically allowed for ΔS > Q/T in an irreversible process? Answer:",
        W5_block="Since entropy can increase by more than Q/T in irreversible processes, could it also increase by less than Q/T in some irreversible process? Answer:",
        W6_allow="Irreversible process: ΔS > Q/T — allowed? Answer:",
        W6_block="Irreversible process: ΔS < Q/T — allowed? Answer:",
        W7_allow=None,
        W7_block=None,
    ),

    dict(
        pair_id="C20", concept="maxwell_demon_information",
        domain="thermodynamics", constraint_type="second_law_information_landauer",
        abstract_rule="Maxwell_demon_cannot_violate_second_law_due_to_Landauer_erasure",
        difficulty="hard", expected_mechanism="entropy_gate",
        wording_mask=["W0_direct", "W2_second_law", "W5_adversarial", "W6_minimal_pair"],
        allowed_core="Maxwell's demon to sort gas molecules by speed using a trap door, while the entropy cost of erasing the demon's memory (Landauer's principle) ensures ΔS_universe ≥ 0",
        blocked_core="Maxwell's demon to operate a trap door that sorts fast molecules from slow ones indefinitely, reducing the entropy of the gas with no compensating entropy increase anywhere",
        physical_direction_allow="maxwell_demon_with_landauer_cost",
        physical_direction_block="maxwell_demon_violating_second_law",
        W4_allow=None,
        W4_block=None,
        W5_allow="Maxwell's demon can sort molecules, but the information it acquires must eventually be erased, producing heat (Landauer's principle). Is it physically allowed for Maxwell's demon to sort molecules if we account for the information cost? Answer:",
        W5_block="Maxwell's demon is just a tiny intelligent observer. Could it sort gas molecules indefinitely, making the gas spontaneously organize itself without any compensating entropy increase anywhere? Answer:",
        W6_allow="Maxwell's demon sorts molecules with full Landauer erasure cost — ΔS_universe ≥ 0 — allowed? Answer:",
        W6_block="Maxwell's demon sorts molecules indefinitely with no information cost or entropy increase anywhere — allowed? Answer:",
        W7_allow=None,
        W7_block=None,
    ),
]


def build_prompt(role: str, concept: dict, wording_key: str) -> dict | None:
    """
    Build a single prompt dict for the given role ('allow'/'block') and wording.
    Returns None if the concept doesn't support this wording.
    """
    core = concept["allowed_core"] if role == "allow" else concept["blocked_core"]
    phys_dir = concept["physical_direction_allow"] if role == "allow" else concept["physical_direction_block"]
    correct_answer = YES_STR if role == "allow" else NO_STR
    incorrect_answer = NO_STR if role == "allow" else YES_STR
    gate_label = role  # "allow" or "block"

    # Resolve template
    if wording_key == "W0_direct":
        prompt_text = f"Is it physically possible for {core}? Answer:"
    elif wording_key == "W1_spontaneous":
        prompt_text = f"Can {core} happen spontaneously, without any external work or energy input? Answer:"
    elif wording_key == "W2_second_law":
        prompt_text = f"Is {core} consistent with the second law of thermodynamics? Answer:"
    elif wording_key == "W3_questioning":
        prompt_text = f"Does the second law of thermodynamics permit {core}? Answer:"
    elif wording_key == "W4_symbolic":
        sym = concept.get(f"W4_{role}")
        if sym is None:
            return None
        prompt_text = f"Is it physically possible for the following to hold? {sym}. Answer:"
    elif wording_key == "W5_adversarial":
        tpl = concept.get(f"W5_{role}")
        if tpl is None:
            return None
        prompt_text = tpl
    elif wording_key == "W6_minimal_pair":
        tpl = concept.get(f"W6_{role}")
        if tpl is None:
            return None
        prompt_text = tpl
    elif wording_key == "W7_experimental":
        tpl = concept.get(f"W7_{role}")
        if tpl is None:
            return None
        prompt_text = tpl
    else:
        return None

    return {
        "prompt": prompt_text,
        "correct_answer": correct_answer,
        "incorrect_answer": incorrect_answer,
        "behaviour": BEHAVIOUR,
        "pair_id": concept["pair_id"],
        "pair_role": role,
        "gate_label": gate_label,
        "physical_direction": phys_dir,
        "concept": concept["concept"],
        "wording_family": wording_key,
        "difficulty": concept["difficulty"],
        "domain": concept["domain"],
        "constraint_type": concept["constraint_type"],
        "abstract_rule": concept["abstract_rule"],
        "expected_mechanism": concept["expected_mechanism"],
        "minimal_pair_id": concept["pair_id"],
    }


def generate_prompts(seed: int = 42) -> list:
    rng = random.Random(seed)
    prompts = []

    for concept in CONCEPTS:
        for wording_key in concept["wording_mask"]:
            for role in ("allow", "block"):
                p = build_prompt(role, concept, wording_key)
                if p is not None:
                    prompts.append(p)

    rng.shuffle(prompts)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = generate_prompts(seed=args.seed)

    with open(out_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    # Summary
    allow_n = sum(1 for p in prompts if p["gate_label"] == "allow")
    block_n = sum(1 for p in prompts if p["gate_label"] == "block")
    by_wording = {}
    by_diff = {}
    for p in prompts:
        by_wording[p["wording_family"]] = by_wording.get(p["wording_family"], 0) + 1
        by_diff[p["difficulty"]] = by_diff.get(p["difficulty"], 0) + 1

    print(f"Generated {len(prompts)} prompts → {out_path}")
    print(f"  allow: {allow_n}  block: {block_n}  (balance: {allow_n/len(prompts):.2f})")
    print(f"  concepts: {len(CONCEPTS)} pairs × up to 8 wordings × 2 roles")
    print(f"  by difficulty: {dict(sorted(by_diff.items()))}")
    print(f"  by wording:")
    for k, v in sorted(by_wording.items()):
        print(f"    {k}: {v}")
    print(f"\nExample prompts:")
    for p in prompts[:3]:
        print(f"  [{p['pair_id']} {p['pair_role']}] {p['prompt'][:90]}...")
        print(f"    correct={p['correct_answer']!r}")
    print(f"\nAnswer tokens: ' Yes' (allow) / ' No' (block)")
    print(f"Next step: python scripts/86_run_entropy_gate_baseline.py --device cuda")


if __name__ == "__main__":
    main()
