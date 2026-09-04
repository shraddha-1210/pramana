"""Layer 6: the five detectors.

Each detector gets a true positive, a true negative, and a boundary case. The
single most important test in the file is
``test_d1_catches_what_d2_cannot`` -- the project's thesis in executable form.
"""

from __future__ import annotations

import pytest

from pramana.attacks.choi_2011 import Choi2011
from pramana.crypto.arbitrator import ArbitrationPanel
from pramana.detectors.base import DetectorKind, Verdict
from pramana.detectors.d1_algebra import D1Algebra
from pramana.detectors.d2_probe_sprt import (
    D2ProbeSprt,
    SprtBounds,
    calibrate_baseline,
    run_sprt,
    theoretical_error_bound,
)
from pramana.detectors.d3_cross_verifier import D3CrossVerifier
from pramana.detectors.d4_ledger import D4Ledger, EntanglementLedger, LedgerTampered
from pramana.detectors.d5_identity_probe import D5IdentityProbe, impersonator_escape_probability
from pramana.engines.base import RoundResult, RoundResults
from pramana.protocol.signing import run_protocol_rounds
from pramana.spec.loader import load_example

PROBE_KEY = b"identity-key-016"


@pytest.fixture(scope="module")
def baseline():
    """The vulnerable scheme every runtime detector is exercised against."""
    return load_example("baseline_bell_aqs")


@pytest.fixture(scope="module")
def honest_run(baseline):
    """An honest run with probes interleaved."""
    return run_protocol_rounds(baseline, rounds=300, seed=11, probe_key=PROBE_KEY)


# --------------------------------------------------------------------------
# THE THESIS.
# --------------------------------------------------------------------------


def test_d1_catches_what_d2_cannot(baseline, honest_run) -> None:
    """The project's thesis, executable.

    A Choi-class forgery **passes verification**, so the runtime statistical
    detector has nothing to see -- there is no anomaly, because the forged pair is
    genuinely valid for the modified message. The static algebraic audit catches
    it, because the vulnerability is in the scheme's structure rather than in the
    traffic.

    If this test ever fails, either the attack stopped working or D2 started
    firing on clean traffic. Both are serious.
    """
    attack = Choi2011().run(baseline, trials=100, seed=7)
    assert attack.verification_accepted is True
    assert attack.mismatch_rate == 0.0

    d2 = D2ProbeSprt().evaluate(baseline, honest_run)
    assert d2.fired is False, "D2 must not fire: the forgery leaves no statistical trace"

    d1 = D1Algebra().evaluate(baseline)
    assert d1.fired is True, "D1 must fire: the scheme is vulnerable by construction"
    assert "X" in d1.evidence["d1a_pauli_witnesses"]


# --------------------------------------------------------------------------
# D1 -- static audit.
# --------------------------------------------------------------------------


def test_d1_is_static_and_needs_no_run(baseline) -> None:
    """D1 takes results=None. It is a pre-deployment audit, not runtime detection."""
    verdict = D1Algebra().evaluate(baseline, None)
    assert verdict.kind is DetectorKind.STATIC
    assert verdict.applicable is True
    assert verdict.confidence == 1.0  # exact algebra, not a statistical score


def test_d1_fires_on_the_baseline_with_the_operator_as_evidence(baseline) -> None:
    verdict = D1Algebra().evaluate(baseline)
    assert verdict.fired is True
    assert verdict.evidence["d1a_pauli_witnesses"] == ["X", "Y", "Z"]
    assert "Choi" in verdict.explanation


def test_d1_fires_on_choi_fixed_because_y_survives() -> None:
    """Choi's published fix still fires: Q = Y. Ledger V-18."""
    verdict = D1Algebra().evaluate(load_example("choi_fixed_aqs"))
    assert verdict.fired is True
    assert verdict.evidence["d1a_pauli_witnesses"] == ["Y"]


def test_d1_clears_the_kim_forgery_free_scheme() -> None:
    """The one scheme that passes the static audit. Kim Corollary 5."""
    verdict = D1Algebra().evaluate(load_example("kim_forgery_free_aqs"))
    assert verdict.fired is False
    assert verdict.evidence["d1a_pauli_witnesses"] == []
    assert verdict.evidence["d1b_forgeable"] is False
    assert "not a proof of security" in verdict.explanation


def test_d1_emits_a_repair_suggestion_naming_the_assistant_operation(baseline) -> None:
    verdict = D1Algebra().evaluate(baseline)
    assert "assistant" in verdict.suggested_fix
    assert "H" in verdict.suggested_fix


def test_d1_reports_the_classical_message_scope_limit() -> None:
    """A zero-witness classical scheme must not read as safe. Ledger V-42."""
    verdict = D1Algebra().evaluate(load_example("pauli_witness_free_aqs"))
    assert "does not imply this scheme is unforgeable" in str(verdict.evidence["scope_note"])


def test_d1_sub_checks_c_and_d_fire_on_the_baseline(baseline) -> None:
    """Positional malleability and swap-test disavowal. Choi section III C."""
    verdict = D1Algebra().evaluate(baseline)
    assert verdict.evidence["d1c_positional_binding"] is False
    assert verdict.evidence["d1d_equality_test"] == "swap"
    assert "PERMUTATION_MALLEABLE" in verdict.explanation
    assert "DISAVOWAL_VIA_SYMMETRIC_STATE" in verdict.explanation


# --------------------------------------------------------------------------
# D2 -- SPRT.
# --------------------------------------------------------------------------


def test_wald_bounds_match_the_formula() -> None:
    """ln((1-beta)/alpha) and ln(beta/(1-alpha)). Derived, not measured."""
    import math

    bounds = SprtBounds.from_error_rates(alpha=0.01, beta=0.02)
    assert bounds.upper == pytest.approx(math.log(0.98 / 0.01))
    assert bounds.lower == pytest.approx(math.log(0.02 / 0.99))
    assert bounds.lower < 0 < bounds.upper


def test_sprt_refuses_a_zero_baseline() -> None:
    """A zero p0 makes the likelihood ratio degenerate. It must raise, not divide.

    This is why the baseline comes from the noise engine and never the Clifford
    engine: the Clifford engine is noiseless and would hand D2 exactly this.
    """
    bounds = SprtBounds.from_error_rates(0.01, 0.01)
    with pytest.raises(ValueError, match="not the Clifford engine"):
        run_sprt([True, False], p0=0.0, p1=0.1, bounds=bounds)


def test_baseline_calibration_uses_the_noise_engine_and_respects_p_min(baseline) -> None:
    """p0 is measured on a noisy channel and floored at the declared p_min."""
    p0 = calibrate_baseline(
        baseline, depolarizing_p=0.0, damping_gamma=0.0, rounds=20, seed=1
    )
    assert p0 == baseline.probes.p_min, "an ideal channel must fall back to the floor"
    assert p0 > 0.0

    noisy = calibrate_baseline(
        baseline, depolarizing_p=0.05, damping_gamma=0.0, rounds=40, seed=1
    )
    assert noisy > baseline.probes.p_min


def test_d2_does_not_fire_on_honest_traffic(baseline, honest_run) -> None:
    """True negative."""
    verdict = D2ProbeSprt().evaluate(baseline, honest_run)
    assert verdict.fired is False


def test_d2_fires_on_a_heavily_disturbed_probe_stream(baseline) -> None:
    """True positive: an error rate far above baseline crosses the upper bound."""
    bounds = SprtBounds.from_error_rates(0.01, 0.01)
    trace = run_sprt([True] * 40, p0=0.02, p1=0.10, bounds=bounds)
    assert trace.decision == "attack"
    assert trace.decided_at is not None


def test_sprt_accepts_the_honest_hypothesis_on_a_clean_stream() -> None:
    """Boundary: a long clean stream must settle on H0, not sit undecided."""
    bounds = SprtBounds.from_error_rates(0.01, 0.01)
    trace = run_sprt([False] * 200, p0=0.02, p1=0.10, bounds=bounds)
    assert trace.decision == "honest"


def test_d2_catches_a_slow_attacker_a_fixed_threshold_would_miss() -> None:
    """The reason for SPRT rather than a threshold.

    An attacker holding the error rate at 1.2x the baseline stays under any fixed
    threshold set at 2x, but evidence accumulates and the sequential test still
    decides -- only later.
    """
    bounds = SprtBounds.from_error_rates(0.05, 0.05)
    p0, slow_rate = 0.05, 0.06
    fixed_threshold = 2 * p0
    assert slow_rate < fixed_threshold, "the premise: a fixed threshold would miss it"

    stream = [i % 100 < 6 for i in range(20_000)]  # 6% error rate
    trace = run_sprt(stream, p0=p0, p1=0.06, bounds=bounds)
    assert trace.decision == "attack"


def test_d2_theory_cross_check_flags_a_poisoned_baseline() -> None:
    """Calibration poisoning defence: a baseline the physics cannot produce.

    Available because the model is analytic. A learned baseline would have
    absorbed the attacker and had nothing to compare against.
    """
    detector = D2ProbeSprt()
    poisoned = detector.check_calibration_against_theory(
        measured_p0=0.40, depolarizing_p=0.01, damping_gamma=0.0
    )
    assert poisoned.fired is True
    assert "attacker was present during calibration" in poisoned.explanation

    plausible = detector.check_calibration_against_theory(
        measured_p0=0.015, depolarizing_p=0.01, damping_gamma=0.005
    )
    assert plausible.fired is False


def test_theoretical_error_bound_is_monotone_and_capped() -> None:
    assert theoretical_error_bound(0.0, 0.0) == 0.0
    assert theoretical_error_bound(0.1, 0.1) == pytest.approx(0.2)
    assert theoretical_error_bound(0.9, 0.9) == 0.5


@pytest.mark.xfail(
    reason=(
        "ASSUMPTION A-3. An attacker present during calibration is absorbed into "
        "the baseline: p0 is measured at the attacker's error rate, so the SPRT "
        "sees nothing anomalous afterwards. This documents the assumption rather "
        "than pretending the detector is unconditional. Expected to fail."
    ),
    strict=True,
)
def test_d2_is_blind_to_an_attacker_present_during_calibration() -> None:
    """Calibration poisoning. Expected failure, documenting A-3."""
    bounds = SprtBounds.from_error_rates(0.01, 0.01)
    attacker_rate = 0.10
    poisoned_p0 = attacker_rate  # the attacker was present when p0 was measured
    stream = [i % 10 == 0 for i in range(500)]  # still 10% errors

    trace = run_sprt(stream, p0=poisoned_p0, p1=0.20, bounds=bounds)
    # This is what we would want to hold. It does not.
    assert trace.decision == "attack"


def test_d2_reports_inapplicable_when_there_are_no_probes(baseline) -> None:
    empty = RoundResults(engine="clifford", seed=0, rounds=())
    verdict = D2ProbeSprt().evaluate(baseline, empty)
    assert verdict.applicable is False
    assert verdict.fired is False
    assert "not the same as finding no attack" in verdict.explanation


# --------------------------------------------------------------------------
# D3 -- cross-verifier.
# --------------------------------------------------------------------------


def _run_with(accepts: list[bool]) -> RoundResults:
    return RoundResults(
        engine="test",
        seed=0,
        rounds=tuple(RoundResult(index=i, accepted=a) for i, a in enumerate(accepts)),
    )


def test_d3_reports_inapplicable_with_a_single_verifier(baseline, honest_run) -> None:
    """Not a silent pass. "Cannot assess" and "found nothing" differ."""
    verdict = D3CrossVerifier({"v1": honest_run}).evaluate(baseline, honest_run)
    assert verdict.applicable is False
    assert verdict.fired is False
    assert "at least two independent verifiers" in verdict.explanation


def test_d3_detects_repudiation_with_two_verifiers(baseline) -> None:
    """True positive: two verifiers disagreeing on the same signature."""
    a = _run_with([True, True, True, True])
    b = _run_with([True, False, True, False])
    verdict = D3CrossVerifier({"v1": a, "v2": b}).evaluate(baseline, a)

    assert verdict.applicable is True
    assert verdict.fired is True
    assert verdict.evidence["disagreement_count"] == 2


def test_d3_does_not_fire_when_verifiers_agree(baseline) -> None:
    a = _run_with([True, True, False])
    b = _run_with([True, True, False])
    assert D3CrossVerifier({"v1": a, "v2": b}).evaluate(baseline, a).fired is False


def test_d3_refuses_to_compare_runs_of_different_lengths(baseline) -> None:
    a, b = _run_with([True, True]), _run_with([True])
    verdict = D3CrossVerifier({"v1": a, "v2": b}).evaluate(baseline, a)
    assert verdict.applicable is False


# --------------------------------------------------------------------------
# D4 -- entanglement ledger.
# --------------------------------------------------------------------------


def test_d4_accepts_a_fresh_run(baseline, honest_run) -> None:
    """True negative, and exact: every index is fresh."""
    verdict = D4Ledger().evaluate(baseline, honest_run)
    assert verdict.fired is False
    assert verdict.confidence == 1.0
    assert verdict.evidence["replayed_count"] == 0


def test_d4_rejects_a_replay_exactly(baseline, honest_run) -> None:
    """True positive, 100%: the same run submitted twice."""
    ledger = EntanglementLedger()
    detector = D4Ledger(ledger=ledger)

    first = detector.evaluate(baseline, honest_run)
    second = detector.evaluate(baseline, honest_run)

    assert first.fired is False
    assert second.fired is True
    assert second.evidence["replayed_count"] > 0
    assert "classical replay" in second.explanation


def test_d4_ledger_integrity_is_mac_protected() -> None:
    """Ledger integrity is derived from the key pool, not assumed."""
    ledger = EntanglementLedger()
    ledger.record((1, 2, 3))
    ledger.check_integrity()  # sealed, so it verifies

    ledger.consumed.add(999)  # tamper without resealing
    with pytest.raises(LedgerTampered, match="does not verify"):
        ledger.check_integrity()


def test_d4_reports_inapplicable_without_a_run(baseline) -> None:
    assert D4Ledger().evaluate(baseline, None).applicable is False


# --------------------------------------------------------------------------
# D5 -- identity-bound probes.
# --------------------------------------------------------------------------


def test_impersonator_escape_probability_is_two_to_the_minus_d() -> None:
    """Derived: an impersonator holds no correlated half, so each probe is a coin."""
    assert impersonator_escape_probability(1) == 0.5
    assert impersonator_escape_probability(32) == 2.0**-32
    assert impersonator_escape_probability(10) == pytest.approx(1 / 1024)


def test_d5_does_not_fire_on_the_legitimate_signer(baseline, honest_run) -> None:
    panel = ArbitrationPanel.build(0xABCDEF, threshold=2, count=4, seed=1)
    verdict = D5IdentityProbe(panel=panel, expected_key=0xABCDEF).evaluate(baseline, honest_run)
    assert verdict.fired is False


def test_d5_detects_an_impersonator(baseline, honest_run) -> None:
    """True positive: uncorrelated probe outcomes."""
    impersonated = RoundResults(
        engine="test",
        seed=0,
        rounds=tuple(
            RoundResult(
                index=r.index,
                accepted=r.accepted,
                is_probe=r.is_probe,
                probe_state=r.probe_state,
                probe_expected=r.probe_expected,
                probe_observed=(1 - (r.probe_expected or 0)) if r.is_probe else None,
            )
            for r in honest_run.rounds
        ),
    )
    panel = ArbitrationPanel.build(0xABCDEF, threshold=2, count=4, seed=1)
    verdict = D5IdentityProbe(panel=panel, expected_key=0xABCDEF).evaluate(
        baseline, impersonated
    )
    assert verdict.fired is True
    assert verdict.evidence["mismatch_rate"] == 1.0


def test_d5_still_fires_when_an_arbitrator_is_malicious(baseline, honest_run) -> None:
    """Non-circularity, proven.

    Expectations come from a t-of-n reconstruction, not from any single
    arbitrator, so a malicious arbitrator cannot suppress its own detection.
    """
    panel = ArbitrationPanel.build(0xABCDEF, threshold=2, count=4, seed=2, dishonest=(3,))
    verdict = D5IdentityProbe(panel=panel, expected_key=0xABCDEF).evaluate(baseline, honest_run)

    assert verdict.fired is True, "a lying arbitrator must not be able to hide"
    assert verdict.evidence["arbitration_suspected"] == [3]
    assert "not circular" in verdict.explanation


def test_d5_reports_inapplicable_without_probes(baseline) -> None:
    empty = RoundResults(engine="t", seed=0, rounds=())
    assert D5IdentityProbe().evaluate(baseline, empty).applicable is False


# --------------------------------------------------------------------------
# Cross-cutting discipline.
# --------------------------------------------------------------------------


def test_every_verdict_carries_an_explanation(baseline, honest_run) -> None:
    """Not optional. A verdict a lab cannot justify is a liability."""
    verdicts = [
        D1Algebra().evaluate(baseline),
        D2ProbeSprt().evaluate(baseline, honest_run),
        D3CrossVerifier({"a": honest_run, "b": honest_run}).evaluate(baseline, honest_run),
        D4Ledger().evaluate(baseline, honest_run),
        D5IdentityProbe().evaluate(baseline, honest_run),
    ]
    for verdict in verdicts:
        assert isinstance(verdict, Verdict)
        assert verdict.explanation.strip()


def test_a_verdict_without_an_explanation_is_refused() -> None:
    """Enforced in the type, not left to reviewer discipline."""
    with pytest.raises(ValueError, match="must be justifiable"):
        Verdict(
            detector="D9",
            kind=DetectorKind.RUNTIME,
            fired=True,
            applicable=True,
            confidence=1.0,
            explanation="   ",
        )


def test_detector_kinds_keep_the_two_axes_separate() -> None:
    """D1 static; D2-D5 runtime. The decision table depends on this."""
    assert D1Algebra().kind is DetectorKind.STATIC
    for detector in (D2ProbeSprt(), D3CrossVerifier(), D4Ledger(), D5IdentityProbe()):
        assert detector.kind is DetectorKind.RUNTIME
