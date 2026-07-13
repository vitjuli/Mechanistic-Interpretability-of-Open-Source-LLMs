"""Unit tests for the particle-pair corpus slicer (src/prompts/make_particle_pairs.py).

Pure-Python tests, no model or data files required.
"""

from src.prompts.make_particle_pairs import PARTICLES, particle_of


def test_particle_of_leading_space():
    # the corpus stores answers like " electron" with a leading space
    assert particle_of({"correct_answer": " electron"}, "correct_answer") == "electron"


def test_particle_of_case_insensitive():
    assert particle_of({"correct_answer": "PROTON"}, "correct_answer") == "proton"


def test_particle_of_substring():
    assert particle_of({"correct_answer": "a neutron is emitted"}, "correct_answer") == "neutron"


def test_particle_of_rejects_non_four():
    # positron/muon are dropped in the K=4 analysis
    assert particle_of({"correct_answer": "muon"}, "correct_answer") is None
    assert particle_of({"correct_answer": "positron"}, "correct_answer") is None


def test_particle_of_missing_key():
    assert particle_of({}, "correct_answer") is None


def test_particles_are_four():
    assert PARTICLES == ["electron", "neutron", "photon", "proton"]
