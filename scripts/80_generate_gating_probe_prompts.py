"""
Generate the gating-probe exploratory benchmark.

8 behaviour families, each testing whether Qwen3-4B has internal "gate" features
that detect physical/mathematical constraint violations:

  A. Thermodynamic spontaneity   (allow / block)
  B. Work-sign / process dir.    (positive / negative work)
  C. Conservation-law check      (allow / block)
  D. Kinematic constraint        (speed increases / decreases)
  E. Vector / math identity      (valid / invalid)
  F. Dimensional analysis        (consistent / inconsistent)
  G. Probability / statistics    (valid / invalid)
  H. Physical boundary cond.     (possible / impossible)

Answer tokens:  ' Yes' (7414) /  ' No' (2308) — single Qwen3 tokens confirmed.

Each concept is a counterfactual minimal pair:
  allowed variant  → correct_answer = ' Yes'
  blocked variant  → correct_answer = ' No'

5 wording families per variant × 2 roles = 10 prompts per concept.
~42 concepts → ~420 prompts total.

Output: data/prompts/gating/gating_probe_v1.jsonl

Usage: python scripts/80_generate_gating_probe_prompts.py
"""

import json, random
from pathlib import Path

OUT  = Path("data/prompts/gating/gating_probe_v1.jsonl")
SEED = 42
random.seed(SEED)

YES_STR = " Yes"
NO_STR  = " No"

# ── Wording templates ─────────────────────────────────────────────────────────
# Applied to each concept's "allowed_core" and "blocked_core".
# {core} = plain core text.  {Core} = sentence-case version.

def _cap(s):
    return s[0].upper() + s[1:] if s else s

WORDING_TEMPLATES = [
    ("W0_direct",    "Is it possible for {core}? Answer:"),
    ("W1_formal",    "Do the fundamental laws of physics or mathematics permit {core}? Answer:"),
    ("W2_student",   "A student argues that {core} can happen. Is the student correct? Answer:"),
    ("W3_scenario",  "Consider this scenario: {core}. Is this physically or mathematically valid? Answer:"),
    ("W4_claim",     "Is {core} a physically or mathematically realizable scenario? Answer:"),
]

# ── Concept definitions ───────────────────────────────────────────────────────
# Each entry: one counterfactual minimal pair.
#   allowed_core: statement that IS physically allowed → answer Yes
#   blocked_core: statement that is NOT allowed        → answer No

CONCEPTS = [

  # ══════════════════════════════════════════════════════════════════════════
  # A. THERMODYNAMIC SPONTANEITY
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="A01", family="A", family_name="thermodynamic_spontaneity",
    concept_key="heat_flow", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="heat_flows_high_to_low_temperature",
    difficulty="easy", expected_mechanism="entropy_gate",
    allowed_core="heat to flow spontaneously from a hot body to a cold body",
    allowed_surface="hot_to_cold", allowed_gate_label="allow",
    blocked_core="heat to flow spontaneously from a cold body to a hot body without any external work",
    blocked_surface="cold_to_hot", blocked_gate_label="block",
  ),
  dict(
    pair_id="A02", family="A", family_name="thermodynamic_spontaneity",
    concept_key="entropy_isolated", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="entropy_cannot_decrease_in_isolated_system",
    difficulty="easy", expected_mechanism="entropy_gate",
    allowed_core="the total entropy of an isolated system to increase during a spontaneous process",
    allowed_surface="entropy_increase", allowed_gate_label="allow",
    blocked_core="the total entropy of an isolated system to decrease during a spontaneous process",
    blocked_surface="entropy_decrease", blocked_gate_label="block",
  ),
  dict(
    pair_id="A03", family="A", family_name="thermodynamic_spontaneity",
    concept_key="gas_mixing", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="mixing_increases_entropy",
    difficulty="medium", expected_mechanism="entropy_gate",
    allowed_core="two different ideal gases to mix spontaneously when the partition between them is removed",
    allowed_surface="mixing", allowed_gate_label="allow",
    blocked_core="a uniform mixture of two gases in a sealed container to spontaneously separate into its pure components without external input",
    blocked_surface="unmixing", blocked_gate_label="block",
  ),
  dict(
    pair_id="A04", family="A", family_name="thermodynamic_spontaneity",
    concept_key="diffusion_direction", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="diffusion_from_high_to_low_concentration",
    difficulty="easy", expected_mechanism="entropy_gate",
    allowed_core="dissolved molecules to diffuse spontaneously from a region of higher concentration to a region of lower concentration",
    allowed_surface="high_to_low_conc", allowed_gate_label="allow",
    blocked_core="dissolved molecules to diffuse spontaneously from a region of lower concentration to a region of higher concentration without external energy input",
    blocked_surface="low_to_high_conc", blocked_gate_label="block",
  ),
  dict(
    pair_id="A05", family="A", family_name="thermodynamic_spontaneity",
    concept_key="carnot_limit", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="real_engine_efficiency_less_than_carnot",
    difficulty="medium", expected_mechanism="carnot_gate",
    allowed_core="a real heat engine operating between reservoirs at 300 K and 600 K to have a thermal efficiency of 40%",
    allowed_surface="below_carnot_limit", allowed_gate_label="allow",
    blocked_core="a real heat engine operating between reservoirs at 300 K and 600 K to have a thermal efficiency of 60%",
    blocked_surface="above_carnot_limit", blocked_gate_label="block",
  ),
  dict(
    pair_id="A06", family="A", family_name="thermodynamic_spontaneity",
    concept_key="perpetual_motion_2", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="cannot_convert_all_heat_to_work_in_cycle",
    difficulty="medium", expected_mechanism="entropy_gate",
    allowed_core="a heat engine to extract heat from a hot reservoir, convert some of it to work, and reject the rest to a cold reservoir",
    allowed_surface="real_engine", allowed_gate_label="allow",
    blocked_core="a machine to continuously extract heat from a single thermal reservoir and convert all of it into useful work with no other effect whatsoever",
    blocked_surface="perpetual_motion_2", blocked_gate_label="block",
  ),
  dict(
    pair_id="A07", family="A", family_name="thermodynamic_spontaneity",
    concept_key="ice_melting", domain="thermodynamics",
    gate_type="allow_block", constraint_type="second_law",
    abstract_rule="phase_transition_direction_depends_on_temperature",
    difficulty="easy", expected_mechanism="entropy_gate",
    allowed_core="ice at atmospheric pressure to melt spontaneously when placed in a room at 25°C",
    allowed_surface="melting_above_0C", allowed_gate_label="allow",
    blocked_core="liquid water at atmospheric pressure to spontaneously freeze into ice in a room at 25°C without removing any heat",
    blocked_surface="freezing_at_25C", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # B. WORK-SIGN / PROCESS DIRECTION
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="B01", family="B", family_name="work_sign",
    concept_key="compression_work_sign", domain="thermodynamics",
    gate_type="allow_block", constraint_type="work_sign_convention",
    abstract_rule="compression_means_negative_work_by_gas",
    difficulty="easy", expected_mechanism="process_direction_gate",
    allowed_core="the work done BY the gas on the surroundings to be negative during isothermal compression of an ideal gas",
    allowed_surface="compression_neg_work", allowed_gate_label="allow",
    blocked_core="the work done BY the gas on the surroundings to be positive during isothermal compression of an ideal gas",
    blocked_surface="compression_pos_work", blocked_gate_label="block",
  ),
  dict(
    pair_id="B02", family="B", family_name="work_sign",
    concept_key="expansion_work_sign", domain="thermodynamics",
    gate_type="allow_block", constraint_type="work_sign_convention",
    abstract_rule="expansion_means_positive_work_by_gas",
    difficulty="easy", expected_mechanism="process_direction_gate",
    allowed_core="the work done BY the gas on the surroundings to be positive during isothermal expansion of an ideal gas",
    allowed_surface="expansion_pos_work", allowed_gate_label="allow",
    blocked_core="the work done BY the gas on the surroundings to be negative during isothermal expansion of an ideal gas",
    blocked_surface="expansion_neg_work", blocked_gate_label="block",
  ),
  dict(
    pair_id="B03", family="B", family_name="work_sign",
    concept_key="compression_temp_adiabatic", domain="thermodynamics",
    gate_type="allow_block", constraint_type="adiabatic_process",
    abstract_rule="adiabatic_compression_raises_temperature",
    difficulty="medium", expected_mechanism="process_direction_gate",
    allowed_core="the temperature of an ideal gas to rise during rapid adiabatic compression",
    allowed_surface="adiabatic_compression_heats", allowed_gate_label="allow",
    blocked_core="the temperature of an ideal gas to fall during rapid adiabatic compression",
    blocked_surface="adiabatic_compression_cools", blocked_gate_label="block",
  ),
  dict(
    pair_id="B04", family="B", family_name="work_sign",
    concept_key="expansion_temp_adiabatic", domain="thermodynamics",
    gate_type="allow_block", constraint_type="adiabatic_process",
    abstract_rule="adiabatic_expansion_cools_gas",
    difficulty="medium", expected_mechanism="process_direction_gate",
    allowed_core="the temperature of an ideal gas to fall during rapid adiabatic expansion",
    allowed_surface="adiabatic_expansion_cools", allowed_gate_label="allow",
    blocked_core="the temperature of an ideal gas to rise during rapid adiabatic expansion",
    blocked_surface="adiabatic_expansion_heats", blocked_gate_label="block",
  ),
  dict(
    pair_id="B05", family="B", family_name="work_sign",
    concept_key="internal_energy_isothermal", domain="thermodynamics",
    gate_type="allow_block", constraint_type="first_law",
    abstract_rule="isothermal_ideal_gas_delta_U_zero",
    difficulty="hard", expected_mechanism="process_direction_gate",
    allowed_core="the internal energy of an ideal gas to remain unchanged during a slow isothermal process",
    allowed_surface="isothermal_delta_U_zero", allowed_gate_label="allow",
    blocked_core="the internal energy of an ideal gas to increase during a slow isothermal expansion",
    blocked_surface="isothermal_delta_U_nonzero", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # C. CONSERVATION LAW (constructed scenarios, not memorised reactions)
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="C01", family="C", family_name="conservation_law",
    concept_key="charge_alpha_decay", domain="nuclear_physics",
    gate_type="allow_block", constraint_type="charge_conservation",
    abstract_rule="charge_is_conserved_in_decay",
    difficulty="medium", expected_mechanism="conservation_gate",
    allowed_core="a nucleus with electric charge +92 to decay by emitting an alpha particle (charge +2) and producing a daughter nucleus with charge +90",
    allowed_surface="charge_balanced", allowed_gate_label="allow",
    blocked_core="a nucleus with electric charge +92 to decay by emitting an alpha particle (charge +2) and producing a daughter nucleus with charge +88",
    blocked_surface="charge_unbalanced", blocked_gate_label="block",
  ),
  dict(
    pair_id="C02", family="C", family_name="conservation_law",
    concept_key="momentum_collision", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="momentum_conservation",
    abstract_rule="total_momentum_conserved_in_isolated_system",
    difficulty="medium", expected_mechanism="conservation_gate",
    allowed_core="two objects in an isolated system to have a total momentum of 15 kg·m/s before and after a collision, given their combined momentum was 15 kg·m/s initially",
    allowed_surface="momentum_conserved", allowed_gate_label="allow",
    blocked_core="two objects in an isolated system to have a total momentum of 20 kg·m/s after a collision, given their combined momentum was 15 kg·m/s initially",
    blocked_surface="momentum_increased", blocked_gate_label="block",
  ),
  dict(
    pair_id="C03", family="C", family_name="conservation_law",
    concept_key="energy_free_fall", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="energy_conservation",
    abstract_rule="KE_gained_equals_PE_lost",
    difficulty="medium", expected_mechanism="conservation_gate",
    allowed_core="a falling object to gain exactly 50 J of kinetic energy after losing exactly 50 J of gravitational potential energy (ignoring air resistance)",
    allowed_surface="energy_conserved", allowed_gate_label="allow",
    blocked_core="a falling object to gain 70 J of kinetic energy after losing only 50 J of gravitational potential energy with no other energy input",
    blocked_surface="energy_created", blocked_gate_label="block",
  ),
  dict(
    pair_id="C04", family="C", family_name="conservation_law",
    concept_key="angular_momentum_no_torque", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="angular_momentum_conservation",
    abstract_rule="angular_momentum_conserved_with_zero_torque",
    difficulty="medium", expected_mechanism="conservation_gate",
    allowed_core="the angular momentum of a spinning object to remain constant when no net external torque acts on it",
    allowed_surface="L_conserved", allowed_gate_label="allow",
    blocked_core="the angular momentum of a spinning object to increase spontaneously when no net external torque acts on it",
    blocked_surface="L_increased_no_torque", blocked_gate_label="block",
  ),
  dict(
    pair_id="C05", family="C", family_name="conservation_law",
    concept_key="baryon_number_decay", domain="nuclear_physics",
    gate_type="allow_block", constraint_type="baryon_conservation",
    abstract_rule="baryon_number_conserved",
    difficulty="hard", expected_mechanism="conservation_gate",
    allowed_core="a neutron (baryon number +1) to decay into a proton (baryon number +1), an electron, and an antineutrino",
    allowed_surface="baryon_balanced", allowed_gate_label="allow",
    blocked_core="a neutron (baryon number +1) to decay into two electrons and a positron with no other particles",
    blocked_surface="baryon_violated", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # D. KINEMATIC CONSTRAINT
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="D01", family="D", family_name="kinematic_constraint",
    concept_key="acc_parallel_speed_increase", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="velocity_acceleration_direction",
    abstract_rule="parallel_acc_increases_speed",
    difficulty="easy", expected_mechanism="direction_gate",
    allowed_core="an object's speed to increase when its acceleration is directed in the same direction as its velocity",
    allowed_surface="acc_parallel_v", allowed_gate_label="allow",
    blocked_core="an object's speed to increase when its acceleration is directed exactly opposite to its velocity",
    blocked_surface="acc_antiparallel_v", blocked_gate_label="block",
  ),
  dict(
    pair_id="D02", family="D", family_name="kinematic_constraint",
    concept_key="acc_antiparallel_speed_decrease", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="velocity_acceleration_direction",
    abstract_rule="antiparallel_acc_decreases_speed",
    difficulty="easy", expected_mechanism="direction_gate",
    allowed_core="an object's speed to decrease when its acceleration is directed exactly opposite to its velocity",
    allowed_surface="acc_antiparallel_decelerates", allowed_gate_label="allow",
    blocked_core="an object's speed to decrease when its acceleration is directed in exactly the same direction as its velocity",
    blocked_surface="acc_parallel_decelerates", blocked_gate_label="block",
  ),
  dict(
    pair_id="D03", family="D", family_name="kinematic_constraint",
    concept_key="rest_positive_acc_positive_velocity", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="kinematics_from_rest",
    abstract_rule="positive_acc_from_rest_gives_positive_velocity",
    difficulty="easy", expected_mechanism="direction_gate",
    allowed_core="an object that starts from rest to have a positive velocity after experiencing a constant positive acceleration for several seconds",
    allowed_surface="rest_pos_acc_pos_vel", allowed_gate_label="allow",
    blocked_core="an object that starts from rest to have a negative velocity after experiencing only constant positive acceleration for several seconds",
    blocked_surface="rest_pos_acc_neg_vel", blocked_gate_label="block",
  ),
  dict(
    pair_id="D04", family="D", family_name="kinematic_constraint",
    concept_key="net_zero_force_constant_vel", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="newtons_first_law",
    abstract_rule="zero_net_force_means_constant_velocity",
    difficulty="medium", expected_mechanism="direction_gate",
    allowed_core="an object's velocity to remain constant (in both magnitude and direction) when the net force on it is exactly zero",
    allowed_surface="F_zero_v_constant", allowed_gate_label="allow",
    blocked_core="an object's velocity to increase in magnitude when the net force on it is exactly zero",
    blocked_surface="F_zero_v_increases", blocked_gate_label="block",
  ),
  dict(
    pair_id="D05", family="D", family_name="kinematic_constraint",
    concept_key="projectile_top_vertical_vel", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="projectile_motion",
    abstract_rule="vertical_velocity_zero_at_max_height",
    difficulty="medium", expected_mechanism="direction_gate",
    allowed_core="the vertical component of velocity of a projectile to be zero at the highest point of its trajectory",
    allowed_surface="vy_zero_at_top", allowed_gate_label="allow",
    blocked_core="the horizontal component of velocity of a projectile to be zero at the highest point of its trajectory (assuming no air resistance)",
    blocked_surface="vx_zero_at_top", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # E. VECTOR / MATH IDENTITY
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="E01", family="E", family_name="vector_math",
    concept_key="curl_of_gradient", domain="vector_calculus",
    gate_type="allow_block", constraint_type="curl_gradient_identity",
    abstract_rule="curl_gradient_always_zero",
    difficulty="medium", expected_mechanism="algebraic_identity_gate",
    allowed_core="the curl of the gradient of any twice-differentiable scalar field to always equal the zero vector",
    allowed_surface="curl_grad_zero", allowed_gate_label="allow",
    blocked_core="the curl of the gradient of a smooth scalar field to be a non-zero vector field in general",
    blocked_surface="curl_grad_nonzero", blocked_gate_label="block",
  ),
  dict(
    pair_id="E02", family="E", family_name="vector_math",
    concept_key="div_of_curl", domain="vector_calculus",
    gate_type="allow_block", constraint_type="div_curl_identity",
    abstract_rule="div_curl_always_zero",
    difficulty="medium", expected_mechanism="algebraic_identity_gate",
    allowed_core="the divergence of the curl of any twice-differentiable vector field to always equal zero",
    allowed_surface="div_curl_zero", allowed_gate_label="allow",
    blocked_core="the divergence of the curl of a smooth vector field to be a non-zero scalar in general",
    blocked_surface="div_curl_nonzero", blocked_gate_label="block",
  ),
  dict(
    pair_id="E03", family="E", family_name="vector_math",
    concept_key="gradient_type", domain="vector_calculus",
    gate_type="allow_block", constraint_type="gradient_output_type",
    abstract_rule="gradient_of_scalar_is_vector",
    difficulty="easy", expected_mechanism="algebraic_identity_gate",
    allowed_core="the gradient of a scalar field to be a vector field",
    allowed_surface="grad_scalar_gives_vector", allowed_gate_label="allow",
    blocked_core="the gradient of a scalar field to itself be a scalar field",
    blocked_surface="grad_scalar_gives_scalar", blocked_gate_label="block",
  ),
  dict(
    pair_id="E04", family="E", family_name="vector_math",
    concept_key="dot_product_commutative", domain="vector_calculus",
    gate_type="allow_block", constraint_type="dot_product_symmetry",
    abstract_rule="dot_product_is_commutative",
    difficulty="easy", expected_mechanism="algebraic_identity_gate",
    allowed_core="the dot product of two vectors A and B to satisfy A·B = B·A",
    allowed_surface="dot_commutes", allowed_gate_label="allow",
    blocked_core="the dot product of two vectors to satisfy A·B = −B·A in general",
    blocked_surface="dot_anticommutes", blocked_gate_label="block",
  ),
  dict(
    pair_id="E05", family="E", family_name="vector_math",
    concept_key="cross_product_anticommutative", domain="vector_calculus",
    gate_type="allow_block", constraint_type="cross_product_symmetry",
    abstract_rule="cross_product_is_anticommutative",
    difficulty="medium", expected_mechanism="algebraic_identity_gate",
    allowed_core="the cross product of two vectors to satisfy A×B = −(B×A)",
    allowed_surface="cross_anticommutes", allowed_gate_label="allow",
    blocked_core="the cross product of two vectors to satisfy A×B = B×A (commutative)",
    blocked_surface="cross_commutes", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # F. DIMENSIONAL ANALYSIS
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="F01", family="F", family_name="dimensional_analysis",
    concept_key="force_ma", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="dimensional_consistency",
    abstract_rule="F_equals_ma_is_dimensionally_consistent",
    difficulty="easy", expected_mechanism="dimensional_gate",
    allowed_core="the equation F = ma (where F is in Newtons, m in kilograms, a in m/s²) to be dimensionally consistent",
    allowed_surface="F_ma_consistent", allowed_gate_label="allow",
    blocked_core="the equation F = ma² (where F is in Newtons, m in kilograms, a in m/s²) to be dimensionally consistent",
    blocked_surface="F_ma_sq_inconsistent", blocked_gate_label="block",
  ),
  dict(
    pair_id="F02", family="F", family_name="dimensional_analysis",
    concept_key="energy_mcsq", domain="relativity",
    gate_type="allow_block", constraint_type="dimensional_consistency",
    abstract_rule="E_equals_mcsq_is_dimensionally_consistent",
    difficulty="easy", expected_mechanism="dimensional_gate",
    allowed_core="the equation E = mc² (where E is in Joules, m in kilograms, c in m/s) to be dimensionally consistent",
    allowed_surface="E_mcsq_consistent", allowed_gate_label="allow",
    blocked_core="the equation E = mc³ (where E is in Joules, m in kilograms, c in m/s) to be dimensionally consistent",
    blocked_surface="E_mccubed_inconsistent", blocked_gate_label="block",
  ),
  dict(
    pair_id="F03", family="F", family_name="dimensional_analysis",
    concept_key="velocity_at", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="dimensional_consistency",
    abstract_rule="v_equals_at_is_dimensionally_consistent",
    difficulty="easy", expected_mechanism="dimensional_gate",
    allowed_core="the kinematic equation v = at (where v is in m/s, a in m/s², t in seconds) to be dimensionally consistent",
    allowed_surface="v_at_consistent", allowed_gate_label="allow",
    blocked_core="the kinematic equation v = at² (where v is in m/s, a in m/s², t in seconds) to be dimensionally consistent",
    blocked_surface="v_atsq_inconsistent", blocked_gate_label="block",
  ),
  dict(
    pair_id="F04", family="F", family_name="dimensional_analysis",
    concept_key="pressure_F_over_A", domain="fluid_mechanics",
    gate_type="allow_block", constraint_type="dimensional_consistency",
    abstract_rule="pressure_equals_force_per_area",
    difficulty="medium", expected_mechanism="dimensional_gate",
    allowed_core="the equation P = F/A (where P is in Pascals, F in Newtons, A in m²) to be dimensionally consistent",
    allowed_surface="P_F_over_A_consistent", allowed_gate_label="allow",
    blocked_core="the equation P = F·A (where P is in Pascals, F in Newtons, A in m²) to be dimensionally consistent",
    blocked_surface="P_F_times_A_inconsistent", blocked_gate_label="block",
  ),
  dict(
    pair_id="F05", family="F", family_name="dimensional_analysis",
    concept_key="acceleration_v_over_t", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="dimensional_consistency",
    abstract_rule="acceleration_equals_velocity_per_time",
    difficulty="easy", expected_mechanism="dimensional_gate",
    allowed_core="the expression a = v/t (where a is in m/s², v in m/s, t in seconds) to be dimensionally consistent",
    allowed_surface="a_v_over_t_consistent", allowed_gate_label="allow",
    blocked_core="the expression a = v·t (where a is in m/s², v in m/s, t in seconds) to be dimensionally consistent",
    blocked_surface="a_v_times_t_inconsistent", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # G. PROBABILITY / STATISTICS CONSTRAINT
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="G01", family="G", family_name="probability_statistics",
    concept_key="probability_bounds_valid", domain="probability",
    gate_type="allow_block", constraint_type="probability_axiom",
    abstract_rule="probability_between_0_and_1",
    difficulty="easy", expected_mechanism="mathematical_bound_gate",
    allowed_core="the probability of an event to be equal to 0.75",
    allowed_surface="prob_valid", allowed_gate_label="allow",
    blocked_core="the probability of an event to be equal to 1.5",
    blocked_surface="prob_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="G02", family="G", family_name="probability_statistics",
    concept_key="variance_nonneg", domain="statistics",
    gate_type="allow_block", constraint_type="variance_nonnegativity",
    abstract_rule="variance_must_be_nonnegative",
    difficulty="easy", expected_mechanism="mathematical_bound_gate",
    allowed_core="the variance of a random variable to be equal to 9",
    allowed_surface="var_valid", allowed_gate_label="allow",
    blocked_core="the variance of a random variable to be equal to −4",
    blocked_surface="var_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="G03", family="G", family_name="probability_statistics",
    concept_key="correlation_bounds", domain="statistics",
    gate_type="allow_block", constraint_type="pearson_correlation_bounds",
    abstract_rule="pearson_r_between_minus1_and_1",
    difficulty="easy", expected_mechanism="mathematical_bound_gate",
    allowed_core="the Pearson correlation coefficient between two variables to be equal to −0.85",
    allowed_surface="corr_valid", allowed_gate_label="allow",
    blocked_core="the Pearson correlation coefficient between two variables to be equal to −1.5",
    blocked_surface="corr_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="G04", family="G", family_name="probability_statistics",
    concept_key="std_dev_nonneg", domain="statistics",
    gate_type="allow_block", constraint_type="std_dev_nonnegativity",
    abstract_rule="standard_deviation_must_be_nonneg",
    difficulty="easy", expected_mechanism="mathematical_bound_gate",
    allowed_core="the standard deviation of a dataset to be equal to 3.5",
    allowed_surface="std_valid", allowed_gate_label="allow",
    blocked_core="the standard deviation of a dataset to be equal to −2",
    blocked_surface="std_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="G05", family="G", family_name="probability_statistics",
    concept_key="prob_sum_to_one", domain="probability",
    gate_type="allow_block", constraint_type="probability_normalization",
    abstract_rule="exhaustive_probabilities_sum_to_one",
    difficulty="medium", expected_mechanism="mathematical_bound_gate",
    allowed_core="a discrete probability distribution over three mutually exclusive exhaustive outcomes to have probabilities 0.2, 0.5, and 0.3",
    allowed_surface="probs_sum_one", allowed_gate_label="allow",
    blocked_core="a discrete probability distribution over three mutually exclusive exhaustive outcomes to have probabilities 0.2, 0.5, and 0.5",
    blocked_surface="probs_sum_not_one", blocked_gate_label="block",
  ),

  # ══════════════════════════════════════════════════════════════════════════
  # H. PHYSICAL BOUNDARY CONDITIONS
  # ══════════════════════════════════════════════════════════════════════════

  dict(
    pair_id="H01", family="H", family_name="physical_boundary",
    concept_key="temperature_kelvin_positive", domain="thermodynamics",
    gate_type="allow_block", constraint_type="absolute_temperature_bound",
    abstract_rule="temperature_in_kelvin_must_be_nonnegative",
    difficulty="easy", expected_mechanism="boundary_gate",
    allowed_core="a thermodynamic system to have a temperature of 300 Kelvin",
    allowed_surface="T_valid", allowed_gate_label="allow",
    blocked_core="a thermodynamic system to have a temperature of −300 Kelvin",
    blocked_surface="T_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="H02", family="H", family_name="physical_boundary",
    concept_key="speed_below_c", domain="relativity",
    gate_type="allow_block", constraint_type="relativistic_speed_limit",
    abstract_rule="massive_objects_cannot_reach_speed_of_light",
    difficulty="medium", expected_mechanism="boundary_gate",
    allowed_core="a proton (which has nonzero rest mass) to travel at 0.99 times the speed of light in a vacuum",
    allowed_surface="v_below_c", allowed_gate_label="allow",
    blocked_core="a proton (which has nonzero rest mass) to travel at exactly the speed of light in a vacuum",
    blocked_surface="v_equals_c", blocked_gate_label="block",
  ),
  dict(
    pair_id="H03", family="H", family_name="physical_boundary",
    concept_key="kinetic_energy_positive", domain="classical_mechanics",
    gate_type="allow_block", constraint_type="kinetic_energy_nonnegativity",
    abstract_rule="kinetic_energy_cannot_be_negative",
    difficulty="easy", expected_mechanism="boundary_gate",
    allowed_core="the kinetic energy of a moving object to be positive",
    allowed_surface="KE_valid", allowed_gate_label="allow",
    blocked_core="the kinetic energy of a moving object to be negative",
    blocked_surface="KE_invalid", blocked_gate_label="block",
  ),
  dict(
    pair_id="H04", family="H", family_name="physical_boundary",
    concept_key="absolute_zero_bound", domain="thermodynamics",
    gate_type="allow_block", constraint_type="third_law",
    abstract_rule="temperature_cannot_reach_absolute_zero_in_finite_steps",
    difficulty="hard", expected_mechanism="boundary_gate",
    allowed_core="a cooling process to bring a system arbitrarily close to absolute zero (0 K) through many steps",
    allowed_surface="approach_zero_K", allowed_gate_label="allow",
    blocked_core="a physical cooling process to bring a system to exactly absolute zero (0 K) in a finite number of steps",
    blocked_surface="reach_zero_K", blocked_gate_label="block",
  ),
  dict(
    pair_id="H05", family="H", family_name="physical_boundary",
    concept_key="efficiency_below_carnot", domain="thermodynamics",
    gate_type="allow_block", constraint_type="carnot_bound",
    abstract_rule="engine_efficiency_cannot_exceed_carnot_efficiency",
    difficulty="medium", expected_mechanism="boundary_gate",
    allowed_core="a heat engine operating between 400 K and 800 K to achieve a thermal efficiency of 45%",
    allowed_surface="eff_below_carnot", allowed_gate_label="allow",
    blocked_core="a heat engine operating between 400 K and 800 K to achieve a thermal efficiency of 60%",
    blocked_surface="eff_above_carnot", blocked_gate_label="block",
  ),
  # Carnot efficiency here = 1 - 400/800 = 0.50. So 45% < 50% allowed, 60% > 50% blocked. ✓
]

# ── Prompt generation ─────────────────────────────────────────────────────────

def generate_prompts(concepts):
    rows = []
    for c in concepts:
        for role, core_key, gate_key, surface_key in [
            ("allowed", "allowed_core", "allowed_gate_label", "allowed_surface"),
            ("blocked", "blocked_core", "blocked_gate_label", "blocked_surface"),
        ]:
            core        = c[core_key]
            gate_label  = c[gate_key]
            surface     = c[surface_key]
            correct     = YES_STR if role == "allowed" else NO_STR
            incorrect   = NO_STR  if role == "allowed" else YES_STR

            for wf, template in WORDING_TEMPLATES:
                prompt = template.format(core=core, Core=_cap(core))
                rows.append({
                    "prompt":            prompt,
                    "correct_answer":    correct,
                    "incorrect_answer":  incorrect,
                    "gate_type":         c["gate_type"],
                    "gate_label":        gate_label,
                    "pair_role":         role,
                    "pair_id":           c["pair_id"],
                    "family":            c["family"],
                    "family_name":       c["family_name"],
                    "concept_key":       c["concept_key"],
                    "domain":            c["domain"],
                    "wording_family":    wf,
                    "constraint_type":   c["constraint_type"],
                    "abstract_rule":     c["abstract_rule"],
                    "surface_direction": surface,
                    "expected_mechanism":c["expected_mechanism"],
                    "difficulty":        c["difficulty"],
                    "experiment_type":   "gating_probe_v1",
                    "multi_token_answer": False,
                })
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_prompts(CONCEPTS)
    random.shuffle(rows)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)

    print(f"Generated {len(rows)} prompts across {len(fams)} families")
    print(f"Saved to {OUT}\n")
    print(f"{'Family':5s} {'Name':30s} {'n_prompts':>10s} {'n_pairs':>8s}")
    print("-" * 60)
    for fam in sorted(fams):
        rr = fams[fam]
        n_pairs = len(set(r["pair_id"] for r in rr))
        print(f"  {fam:3s}  {rr[0]['family_name']:30s}  {len(rr):>8d}  {n_pairs:>7d}")
    print("-" * 60)
    print(f"  {'TOT':3s}  {'':30s}  {len(rows):>8d}  {len(set(r['pair_id'] for r in rows)):>7d}")


if __name__ == "__main__":
    main()
