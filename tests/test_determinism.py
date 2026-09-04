"""Rule 6: every simulation entry point is reproducible under an explicit seed.

Two runs with the same seed produce identical output; two runs with different
seeds do not. Cheap to check, and it rules out a class of bug that is otherwise
brutal to debug in a statistical system.

**Scope of the claim.** Reproducibility is asserted within a single environment
with pinned library versions. Stim's own documentation states that seeding
reproduces results only for the same Stim version on the same machine, so
cross-platform byte-identity is not claimed anywhere in this project. Ledger V-12.
"""

from __future__ import annotations

import pytest

from pramana.engines.clifford import CliffordEngine
from pramana.engines.noise import NoiseEngine
from pramana.protocol.teleportation import PAULI_EIGENSTATES, teleport
from pramana.spec.loader import load_example

ROUNDS = 100
SEED = 2024


@pytest.fixture(scope="module")
def spec():
    """The baseline scheme, used identically by every engine under test."""
    return load_example("baseline_bell_aqs")


def _engines():
    """Every engine that consumes randomness, with a noisy configuration too.

    The noiseless case can be reproducible for the trivial reason that nothing
    varies, so a genuinely stochastic configuration is included as well.
    """
    return [
        ("clifford", CliffordEngine()),
        ("noise (ideal)", NoiseEngine()),
        ("noise (depolarizing)", NoiseEngine(depolarizing_p=0.05)),
        ("noise (damping)", NoiseEngine(damping_gamma=0.10)),
    ]


@pytest.mark.parametrize("label,engine", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_same_seed_reproduces_identical_output(label: str, engine, spec) -> None:
    """Two runs at one seed are bit-identical, round for round."""
    first = engine.run_rounds(spec, rounds=ROUNDS, seed=SEED)
    second = engine.run_rounds(spec, rounds=ROUNDS, seed=SEED)

    assert first.fingerprint() == second.fingerprint(), f"{label} was not reproducible"
    assert first.accept_rate == second.accept_rate


@pytest.mark.parametrize("label,engine", _engines(), ids=lambda v: v if isinstance(v, str) else "")
def test_different_seeds_produce_different_output(label: str, engine, spec) -> None:
    """A different seed must actually change the run.

    Guards the failure mode where a seed is accepted and then ignored, which
    would make the reproducibility test above pass vacuously.
    """
    first = engine.run_rounds(spec, rounds=ROUNDS, seed=SEED)
    other = engine.run_rounds(spec, rounds=ROUNDS, seed=SEED + 1)

    assert first.fingerprint() != other.fingerprint(), f"{label} ignored its seed"


def test_teleportation_is_reproducible_under_a_seed() -> None:
    """The teleportation primitive itself, not only the engines wrapping it."""
    for state in PAULI_EIGENSTATES:
        first = teleport(state, seed=99)
        second = teleport(state, seed=99)
        assert first == second


def test_teleportation_seed_actually_varies_the_outcome() -> None:
    """Across seeds a state must reach more than one Bell outcome."""
    outcomes = {teleport("|0>", seed=s).outcome for s in range(50)}
    assert len(outcomes) == 4, f"expected all four Bell outcomes across seeds, saw {outcomes}"


def test_engine_results_record_the_seed_that_produced_them(spec) -> None:
    """Every result carries its seed, so any number in the UI is reproducible."""
    for _label, engine in _engines():
        results = engine.run_rounds(spec, rounds=5, seed=SEED)
        assert results.seed == SEED
        assert results.engine
