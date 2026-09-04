"""Layer 2 exit gate: teleportation, the correction table, and the two engines.

The teleportation test is the hard gate for the whole project. Everything
downstream is meaningless if the correction convention is wrong, so the table is
*established* here rather than quoted, and the test that establishes it is the
one that would fail if the convention were wrong.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from pramana.engines.clifford import CliffordEngine
from pramana.engines.noise import NoiseEngine
from pramana.protocol.teleportation import (
    CORRECTION_TABLE,
    PAULI_EIGENSTATES,
    PAULI_NAMES,
    anticommutes,
    determine_correction_table,
    expected_bit,
    teleport,
)
from pramana.spec.loader import load_example

ALL_OUTCOMES = ((0, 0), (0, 1), (1, 0), (1, 1))
SHOTS = 200


@pytest.fixture(scope="module")
def spec():
    """The baseline scheme; its geometry is what the engines are exercised on."""
    return load_example("baseline_bell_aqs")


# --------------------------------------------------------------------------
# THE GATE.
# --------------------------------------------------------------------------


def test_teleportation_recovers_all_six_pauli_eigenstates() -> None:
    """Every Pauli eigenstate teleports with fidelity 1, for every Bell outcome.

    **The hard gate.** Build plan Layer 2: "Do not proceed past this gate.
    Everything downstream is meaningless if teleportation is wrong."

    Strengthened from the plan's one-shot-per-state version, which exercises only
    one of the four measurement outcomes and so passes roughly three times in four
    with a table that is wrong on a single entry (ledger Q-4). Here every state is
    run over many seeds and the test additionally asserts that all four outcomes
    actually occurred, so a wrong entry cannot hide behind an unsampled branch.

    Recovery is checked with an exact stabilizer expectation, not by sampling, so
    each shot is a statement about the state rather than about a finite sample.
    """
    for state in PAULI_EIGENSTATES:
        outcomes_seen: Counter[tuple[int, int]] = Counter()
        for seed in range(SHOTS):
            shot = teleport(state, seed=seed)
            outcomes_seen[shot.outcome] += 1
            assert shot.recovered, (
                f"{state} was not recovered on Bell outcome {shot.outcome} "
                f"under correction {shot.correction!r} (seed {seed})"
            )
        missing = set(ALL_OUTCOMES) - set(outcomes_seen)
        assert not missing, f"{state}: Bell outcomes never exercised: {sorted(missing)}"


def test_correction_table_is_uniquely_determined() -> None:
    """Re-derive the table from scratch; it must be unique and match the constant.

    This is the empirical determination the build plan calls for, run as a test:
    for each Bell outcome, try all four Paulis and keep those that recover every
    one of the six eigenstates. Exactly one survives per outcome, so the
    convention is forced rather than chosen.

    If this fails, the recorded constant is wrong. Fix the constant, never this
    test.
    """
    determined = determine_correction_table(shots_per_state=SHOTS)

    assert set(determined) == set(ALL_OUTCOMES), "not every Bell outcome was sampled"
    for outcome, valid in determined.items():
        assert len(valid) == 1, (
            f"outcome {outcome} admits {len(valid)} corrections {valid}; "
            "the convention is not uniquely determined"
        )
    assert {outcome: valid[0] for outcome, valid in determined.items()} == CORRECTION_TABLE


@pytest.mark.parametrize("swap_with", ["X", "Y", "Z"])
def test_a_wrong_correction_table_is_caught(swap_with: str) -> None:
    """The gate test has power: corrupting one entry makes recovery fail.

    Without this, a test that passed vacuously would look identical to one that
    passed for the right reason.
    """
    corrupted = dict(CORRECTION_TABLE)
    corrupted[(0, 0)] = swap_with  # the identity entry is unambiguous to corrupt

    failures = [
        state
        for state in PAULI_EIGENSTATES
        for seed in range(SHOTS)
        if not teleport(state, seed=seed, table=corrupted).recovered
    ]
    assert failures, f"corrupting the (0,0) entry to {swap_with} was not detected"


def test_correction_table_covers_every_outcome_with_a_known_pauli() -> None:
    """The table is total over Bell outcomes and names only registry Paulis."""
    assert set(CORRECTION_TABLE) == set(ALL_OUTCOMES)
    assert set(CORRECTION_TABLE.values()) == set(PAULI_NAMES)


# --------------------------------------------------------------------------
# Correction algebra used by the measurement-based engine.
# --------------------------------------------------------------------------


def test_anticommutation_of_corrections_with_observables() -> None:
    """Distinct non-identity Paulis anticommute; identity commutes with all.

    The noise engine folds the correction into the expected classical bit using
    this, so it has to be right independently of any simulator.
    """
    for correction in PAULI_NAMES:
        for observable in ("X", "Y", "Z"):
            expected = correction != "I" and correction != observable
            assert anticommutes(correction, observable) is expected
    assert anticommutes("X", "I") is False


def test_expected_bit_mapping() -> None:
    """+1 eigenvalue measures as 0, -1 measures as 1."""
    assert expected_bit(+1) == 0
    assert expected_bit(-1) == 1


# --------------------------------------------------------------------------
# Honest acceptance. Ledger V-01.
# --------------------------------------------------------------------------


def test_honest_acceptance_is_exactly_one_over_10k_clifford_rounds(spec) -> None:
    """V-01: an honest round accepts with probability 1 on a noiseless channel.

    Exactly 1.0, not approximately: noiseless Clifford simulation is
    deterministic given the correction table, so any shortfall is a bug rather
    than sampling error. No tolerance is appropriate here.
    """
    results = CliffordEngine().run_rounds(spec, rounds=10_000, seed=42)

    assert len(results.rounds) == 10_000
    assert results.accept_rate == 1.0
    assert all(r.mismatch_positions == () for r in results.rounds)


def test_every_round_consumes_fresh_entangled_pairs(spec) -> None:
    """Pair indices are never reused. D4 turns this into an exact replay check."""
    results = CliffordEngine().run_rounds(spec, rounds=50, seed=3)
    consumed = [i for r in results.rounds for i in r.pair_indices]

    assert len(consumed) == len(set(consumed)), "an entangled-pair index was reused"
    assert consumed == sorted(consumed)
    expected_per_round = spec.signature.length_qubits * spec.entanglement.pairs_per_signature_qubit
    assert len(consumed) == 50 * expected_per_round


def test_round_records_carry_mismatch_detail_not_just_a_boolean(spec) -> None:
    """Detectors need positions, so the field must exist and be well-formed."""
    results = CliffordEngine().run_rounds(spec, rounds=5, seed=1)
    for r in results.rounds:
        assert r.accepted == (r.mismatch_positions == ())
        assert len(r.outcomes) == spec.signature.length_qubits
        assert len(r.corrections) == spec.signature.length_qubits
        assert all(c in PAULI_NAMES for c in r.corrections)


# --------------------------------------------------------------------------
# Noise engine. Ledger V-32.
# --------------------------------------------------------------------------


def test_noiseless_noise_engine_accepts_every_round(spec) -> None:
    """With both parameters zero the channel is ideal and acceptance is exactly 1.

    This is the achievable form of the build plan's "matches the Clifford engine
    exactly" test. The two engines cannot produce identical outcome *sequences* --
    they draw from unrelated RNG streams -- so what is asserted is agreement on
    the deterministic quantities. Ledger Q-2.
    """
    noise = NoiseEngine().run_rounds(spec, rounds=200, seed=11)
    clifford = CliffordEngine().run_rounds(spec, rounds=200, seed=11)

    assert noise.accept_rate == 1.0
    assert noise.accept_rate == clifford.accept_rate
    assert all(r.mismatch_positions == () for r in noise.rounds)


def test_acceptance_falls_monotonically_as_depolarizing_noise_rises(spec) -> None:
    """More depolarizing noise never helps.

    Only the direction is asserted, and only the p = 0 endpoint is asserted
    exactly. The quantitative accept-rate curve as a function of p has *not* been
    derived analytically for this circuit, so no intermediate value is claimed
    here -- a snapshot of the current output would violate Rule 3. Ledger V-34.
    """
    rates = [
        NoiseEngine(depolarizing_p=p).run_rounds(spec, rounds=200, seed=7).accept_rate
        for p in (0.0, 0.01, 0.05, 0.20)
    ]

    assert rates[0] == 1.0
    assert all(later <= earlier for earlier, later in pairwise(rates)), rates
    assert rates[-1] < rates[0]


def test_amplitude_damping_also_degrades_acceptance(spec) -> None:
    """Damping is a distinct channel and must degrade acceptance on its own."""
    clean = NoiseEngine().run_rounds(spec, rounds=200, seed=7).accept_rate
    damped = NoiseEngine(damping_gamma=0.15).run_rounds(spec, rounds=200, seed=7).accept_rate

    assert clean == 1.0
    assert damped < clean


@pytest.mark.parametrize(
    ("depolarizing_p", "damping_gamma"),
    [(-0.1, 0.0), (1.5, 0.0), (0.0, -0.1), (0.0, 1.5)],
)
def test_noise_parameters_outside_zero_to_one_are_rejected(
    depolarizing_p: float, damping_gamma: float
) -> None:
    """A probability outside [0, 1] is a specification error, not a clamp."""
    with pytest.raises(ValueError, match="must be in"):
        NoiseEngine(depolarizing_p=depolarizing_p, damping_gamma=damping_gamma)


# --------------------------------------------------------------------------
# LAYER 3 — Signing, verification, and probes.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def choi_spec():
    """The Choi-fixed scheme, for testing (U,V)-type encryption."""
    return load_example("choi_fixed_aqs")


@pytest.fixture(scope="module")
def witness_free_spec():
    """The Pauli-witness-free scheme, for testing composite factors."""
    return load_example("pauli_witness_free_aqs")


@pytest.fixture(scope="module")
def kim_spec():
    """The Kim forgery-free scheme -- non-Clifford, cannot be signed."""
    return load_example("kim_forgery_free_aqs")


def test_signing_produces_a_valid_signature_that_verification_accepts(spec) -> None:
    """V-42: every honestly signed message verifies on a noiseless channel.

    10k rounds, exact. Any shortfall is a bug in the signing/verification path,
    not sampling noise. This is the Layer 3 analogue of V-01.
    """
    from pramana.protocol.signing import run_protocol_rounds

    results = run_protocol_rounds(spec, rounds=10_000, seed=42)

    assert len(results.rounds) == 10_000
    assert results.accept_rate == 1.0
    assert all(r.accepted for r in results.rounds)


def test_signing_with_uv_type_encryption_also_accepts(choi_spec) -> None:
    """Signing with (I,H)-type encryption verifies on a noiseless channel.

    Tests that the factor application and inversion are correct for a non-
    trivial right factor.
    """
    from pramana.protocol.signing import run_protocol_rounds

    results = run_protocol_rounds(choi_spec, rounds=1_000, seed=99)

    assert results.accept_rate == 1.0


def test_signing_with_composite_factor_also_accepts(witness_free_spec) -> None:
    """Signing with [S, H] right factor verifies on a noiseless channel.

    Tests that composite factor sequences are applied and inverted correctly.
    """
    from pramana.protocol.signing import run_protocol_rounds

    results = run_protocol_rounds(witness_free_spec, rounds=1_000, seed=77)

    assert results.accept_rate == 1.0


def test_signing_twice_with_different_seeds_produces_different_signatures(spec) -> None:
    """V-45: different seeds yield different corrections and pair indices.

    What makes this true: each signing run consumes entangled pairs from a
    monotonically advancing counter (V-36), and the Bell-measurement outcomes
    come from a different RNG stream. The pair index alone is sufficient --
    physics forbids reuse.
    """
    from pramana.protocol.signing import run_protocol_rounds

    r1 = run_protocol_rounds(spec, rounds=10, seed=1)
    r2 = run_protocol_rounds(spec, rounds=10, seed=2)

    # Corrections should differ (different RNG streams).
    corrections_1 = tuple(r.corrections for r in r1.rounds)
    corrections_2 = tuple(r.corrections for r in r2.rounds)
    assert corrections_1 != corrections_2, (
        "two runs with different seeds produced identical corrections"
    )

    # Outcomes should differ.
    outcomes_1 = tuple(r.outcomes for r in r1.rounds)
    outcomes_2 = tuple(r.outcomes for r in r2.rounds)
    assert outcomes_1 != outcomes_2, (
        "two runs with different seeds produced identical outcomes"
    )


def test_probe_outcomes_match_expectation_with_probability_one(spec) -> None:
    """V-44: honest probes match on a noiseless Clifford channel.

    This is a correctness test only, not a calibration source. It will not
    be passed to D2.
    """
    from pramana.protocol.signing import run_protocol_rounds

    probe_key = b"test_probe_key_material_32_bytes!"
    results = run_protocol_rounds(spec, rounds=200, seed=42, probe_key=probe_key)

    probes = [r for r in results.rounds if r.is_probe]
    assert len(probes) > 0, "no probes were scheduled"
    assert len(probes) == min(spec.probes.rounds_per_signature, 200)

    for r in probes:
        assert r.probe_state is not None
        assert r.probe_expected is not None
        assert r.probe_observed is not None
        assert r.probe_observed == r.probe_expected, (
            f"probe at round {r.index} for state {r.probe_state}: "
            f"expected {r.probe_expected}, got {r.probe_observed}"
        )

    assert results.probe_mismatch_rate == 0.0


def test_probe_positions_are_unpredictable_without_key() -> None:
    """V-43: probe schedule is a deterministic function of key material.

    Given the probe schedule for one key, an attacker without the key
    cannot predict it. Tested by checking that different keys produce
    different schedules over many trials, and that the schedules are
    deterministic (same key + same parameters = same schedule).
    """
    from pramana.protocol.probes import derive_probe_schedule

    total = 500
    count = 50

    # Same key, same parameters -> same schedule.
    key = b"shared_secret_key_for_probes_32b!"
    s1 = derive_probe_schedule(key, total, count)
    s2 = derive_probe_schedule(key, total, count)
    assert s1 == s2, "same key produced different schedules"

    # Different keys -> different schedules.
    schedules: set[frozenset[int]] = set()
    for i in range(100):
        k = f"key_{i:04d}_________________________________".encode()[:32]
        schedules.add(derive_probe_schedule(k, total, count))

    # With 100 random keys and 50-of-500 positions, all 100 schedules should
    # be distinct. The probability of a collision is negligible.
    assert len(schedules) == 100, (
        f"only {len(schedules)} distinct schedules from 100 different keys"
    )


def test_probe_schedule_count_is_exact() -> None:
    """The schedule contains exactly the requested number of probes."""
    from pramana.protocol.probes import derive_probe_schedule

    key = b"test_key_for_count_checking_32b!"
    for count in (0, 1, 10, 50, 100):
        s = derive_probe_schedule(key, 100, count)
        assert len(s) == count


def test_probe_schedule_rejects_more_probes_than_rounds() -> None:
    """Cannot schedule more probes than total rounds."""
    from pramana.protocol.probes import derive_probe_schedule

    with pytest.raises(ValueError, match="exceeds total_rounds"):
        derive_probe_schedule(b"key", 10, 20)


def test_non_clifford_scheme_refuses_to_sign(kim_spec) -> None:
    """V-40 at Layer 3: a non-Clifford scheme cannot be signed.

    kim_forgery_free_aqs declares W_kim_forgery_free, a non-Clifford
    assistant unitary. The signing path cannot simulate it. Static analysis
    (D1) still applies.
    """
    from pramana.protocol.signing import run_protocol_rounds

    with pytest.raises(NotImplementedError, match="non-Clifford"):
        run_protocol_rounds(kim_spec, rounds=1, seed=42)


def test_three_rotation_scheme_refuses_to_sign() -> None:
    """Q-14: a scheme with rotation_count >= 3 cannot be signed at Layer 3.

    The rotation operators for 3+ rotations are not modelled: sigma_x and
    sigma_z already generate the single-qubit Pauli group mod phase, so no
    third independent Pauli rotation exists. Do not guess.
    """
    from pramana.protocol.signing import _reject_three_or_more_rotations

    # kim_forgery_free_aqs has rotation_count=3, but also non-Clifford.
    # Test the rotation check directly.
    spec = load_example("kim_forgery_free_aqs")
    with pytest.raises(NotImplementedError, match="rotation_count"):
        _reject_three_or_more_rotations(spec)


def test_non_probe_rounds_have_no_probe_fields(spec) -> None:
    """Non-probe rounds do not populate probe fields."""
    from pramana.protocol.signing import run_protocol_rounds

    results = run_protocol_rounds(spec, rounds=10, seed=42)

    for r in results.rounds:
        assert r.is_probe is False
        assert r.probe_state is None
        assert r.probe_expected is None
        assert r.probe_observed is None


def test_all_rounds_consume_fresh_pairs_at_layer_3(spec) -> None:
    """V-36 at Layer 3: pair indices are strictly increasing, never reused."""
    from pramana.protocol.signing import run_protocol_rounds

    results = run_protocol_rounds(spec, rounds=50, seed=3)
    consumed = [i for r in results.rounds for i in r.pair_indices]

    assert len(consumed) == len(set(consumed)), "pair index reused"
    assert consumed == sorted(consumed), "pair indices not monotonically increasing"

