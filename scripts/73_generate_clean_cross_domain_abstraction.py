"""
Generate clean cross-domain v2 held-out prompts for physics_intensive_extensive_v1.

Key design decision: all four wording families use formally identical scaling
operations, with no vague language ("scale up", "grow", "expand"). The operation
is always one of:
  W1_duplicate  — exact duplication, local ratios unchanged
  W2_combine    — two identical systems merged, local quantities preserved
  W3_split      — split into two equal halves, per-unit quantities preserved
  W4_ratio_fixed — double size with all ratios/rates/densities held constant

Correct answer:
  ' extensive' — if the property doubles under the scaling operation
  ' intensive' — if the property stays the same

Saves: data/prompts/abstraction/clean_cross_domain_v2.jsonl

Usage:
    python scripts/73_generate_clean_cross_domain_abstraction.py
"""

import json
import random
from pathlib import Path

OUT_PATH = Path("data/prompts/abstraction/clean_cross_domain_v2.jsonl")
SEED = 42
random.seed(SEED)

# ── Property definitions ──────────────────────────────────────────────────────
# Each entry: (property_key, property_phrase, domain, system_singular, system_plural, cls)
# cls: "intensive" (stays same) or "extensive" (doubles)

PROPERTIES = [
    # ── ECONOMICS ─────────────────────────────────────────────────────────────
    # intensive
    ("interest_rate",      "the interest rate",            "economics",       "loan portfolio",    "loan portfolios",    "intensive"),
    ("profit_margin",      "the profit margin",            "economics",       "business",          "businesses",         "intensive"),
    ("price_per_unit",     "the price per unit",           "economics",       "market",            "markets",            "intensive"),
    ("inflation_rate",     "the inflation rate",           "economics",       "economy",           "economies",          "intensive"),
    ("tax_rate",           "the tax rate",                 "economics",       "fiscal system",     "fiscal systems",     "intensive"),
    ("exchange_rate",      "the exchange rate",            "economics",       "currency market",   "currency markets",   "intensive"),
    # extensive
    ("total_revenue",      "the total revenue",            "economics",       "business",          "businesses",         "extensive"),
    ("total_cost",         "the total cost",               "economics",       "production system", "production systems", "extensive"),
    ("gdp",                "the gross domestic product",   "economics",       "economy",           "economies",          "extensive"),
    ("total_profit",       "the total profit",             "economics",       "firm",              "firms",              "extensive"),
    ("total_assets",       "the total assets",             "economics",       "portfolio",         "portfolios",         "extensive"),

    # ── BIOLOGY ───────────────────────────────────────────────────────────────
    # intensive
    ("metabolic_rate_per_kg",  "the metabolic rate per kilogram",  "biology",  "animal colony",    "animal colonies",   "intensive"),
    ("population_density",     "the population density",           "biology",  "habitat",          "habitats",          "intensive"),
    ("per_capita_growth_rate", "the per-capita growth rate",       "biology",  "population",       "populations",       "intensive"),
    ("mortality_rate",         "the mortality rate per individual", "biology",  "population",       "populations",       "intensive"),
    # extensive
    ("total_biomass",          "the total biomass",                "biology",  "ecosystem",        "ecosystems",        "extensive"),
    ("total_population",       "the total population count",       "biology",  "habitat",          "habitats",          "extensive"),
    ("total_energy_output",    "the total energy output",          "biology",  "organism colony",  "organism colonies", "extensive"),
    ("total_oxygen_consumed",  "the total oxygen consumed per day","biology",  "organism colony",  "organism colonies", "extensive"),

    # ── STATISTICS ────────────────────────────────────────────────────────────
    # intensive
    ("mean",               "the mean (average)",           "statistics",      "dataset",           "datasets",           "intensive"),
    ("variance",           "the variance",                 "statistics",      "dataset",           "datasets",           "intensive"),
    ("median",             "the median",                   "statistics",      "dataset",           "datasets",           "intensive"),
    # extensive
    ("sum",                "the sum",                      "statistics",      "dataset",           "datasets",           "extensive"),
    ("total_sq_dev",       "the total squared deviation from the mean", "statistics", "dataset", "datasets",           "extensive"),
    ("sample_count",       "the sample count",             "statistics",      "dataset",           "datasets",           "extensive"),

    # ── INFORMATION THEORY ────────────────────────────────────────────────────
    # intensive
    ("entropy_rate",       "the entropy rate (bits per symbol)", "information theory", "data stream", "data streams",   "intensive"),
    ("compression_ratio",  "the compression ratio",        "information theory",      "data file",    "data files",     "intensive"),
    ("bit_error_rate",     "the bit error rate",           "information theory",      "communication channel", "communication channels", "intensive"),
    # extensive
    ("total_file_size",    "the total file size",          "information theory",      "digital archive", "digital archives", "extensive"),
    ("total_bits",         "the total number of bits",     "information theory",      "message",       "messages",       "extensive"),
    ("total_information",  "the total Shannon information","information theory",      "data stream",   "data streams",   "extensive"),
]

# ── Wording templates ─────────────────────────────────────────────────────────
# {prop} = property_phrase, {sys} = system_singular, {sysp} = system_plural
# Answer: ' extensive' if property doubles, ' intensive' if it stays the same.

WORDING_FAMILIES = {
    "W1_duplicate": (
        "Suppose a {sys} is exactly duplicated — two independent copies are created, "
        "with all internal ratios, densities, and per-unit quantities left unchanged. "
        "Does {prop} of the two-copy system equal twice {prop} of the original? Answer:"
    ),
    "W2_combine": (
        "Two separate but physically identical {sysp} are merged into a single combined system. "
        "Every local ratio and per-unit quantity remains the same as before merging. "
        "Does {prop} of the merged system equal twice {prop} of a single {sys}? Answer:"
    ),
    "W3_split": (
        "A {sys} is divided into two equal halves. "
        "Each half is a perfect smaller replica, with all per-unit quantities and local ratios "
        "preserved exactly from the original. "
        "Does {prop} of each half equal exactly half of {prop} of the full {sys}? Answer:"
    ),
    "W4_ratio_fixed": (
        "The size of a {sys} is doubled while strictly preserving every internal ratio, "
        "concentration, rate, and density. "
        "Does {prop} also double as a result? Answer:"
    ),
}

# Answer logic: for all four wordings the question is "does the property double?"
# extensive = YES (doubles) → token ' extensive'
# intensive = NO (stays same) → token ' intensive'

INT_TOKEN_STR = " intensive"
EXT_TOKEN_STR = " extensive"


def make_prompts():
    rows = []
    for (prop_key, prop_phrase, domain, sys_sg, sys_pl, cls) in PROPERTIES:
        correct   = INT_TOKEN_STR if cls == "intensive" else EXT_TOKEN_STR
        incorrect = EXT_TOKEN_STR if cls == "intensive" else INT_TOKEN_STR

        for wf, template in WORDING_FAMILIES.items():
            prompt = template.format(prop=prop_phrase, sys=sys_sg, sysp=sys_pl)
            rows.append({
                "prompt":             prompt,
                "correct_answer":     correct,
                "incorrect_answer":   incorrect,
                "abstraction_class":  cls,
                "property":           prop_key,
                "property_phrase":    prop_phrase,
                "domain":             domain,
                "wording_family":     wf,
                "experiment_type":    "clean_cross_domain_v2",
                "scaling_operation":  (
                    "duplicate_without_changing_local_ratios"   if wf == "W1_duplicate" else
                    "combine_two_identical_systems_local_preserved" if wf == "W2_combine" else
                    "split_into_two_equal_halves_ratios_preserved"  if wf == "W3_split"  else
                    "double_size_all_ratios_fixed"
                ),
            })

    random.shuffle(rows)
    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = make_prompts()

    n_int = sum(1 for r in rows if r["abstraction_class"] == "intensive")
    n_ext = sum(1 for r in rows if r["abstraction_class"] == "extensive")
    print(f"Generated {len(rows)} prompts: {n_int} intensive, {n_ext} extensive")
    print(f"Properties: {len(PROPERTIES)}, wording families: {len(WORDING_FAMILIES)}")
    print(f"Domains: {sorted(set(r['domain'] for r in rows))}")

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved to {OUT_PATH}")
    print("\nSample prompts:")
    for r in rows[:3]:
        print(f"\n[{r['domain']} / {r['property']} / {r['wording_family']} / {r['abstraction_class']}]")
        print(r["prompt"][:200])


if __name__ == "__main__":
    main()
