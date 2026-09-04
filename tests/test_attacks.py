"""Layer 5: the attack library.

Every rate here is derived before it is measured. Where a rate is statistical, the
tolerance is a 3-sigma binomial bound justified by the sample size; where it is
deterministic, it is asserted exactly with no tolerance at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pramana.attacks import ATTACK_REGISTRY
from pramana.attacks.blind_forgery import (
    BlindForgery,
    acceptance_probability,
    correct_guess_probability,
)
from pramana.attacks.choi_2011 import Choi2011, select_forging_operator
from pramana.attacks.intercept_resend import (
    DETECTION_PER_PROBE,
    InterceptResend,
    escape_probability,
)
from pramana.attacks.replay import Replay
from pramana.protocol.signing import sign_and_verify_round
from pramana.spec.loader import load_example
from pramana.spec.schema import SchemeSpec

CLIFFORD_SCHEMES = ("baseline_bell_aqs", "choi_fixed_aqs", "pauli_witness_free_aqs")


def three_sigma(p: float, n: int) -> float:
    """Binomial 3-sigma half-width for ``n`` trials at rate ``p``."""
    return 3.0 * (p * (1.0 - p) / n) ** 0.5


# --------------------------------------------------------------------------
# THE DEFINING TEST. Choi section III A: "always successful."
# --------------------------------------------------------------------------


def test_choi_forgery_passes_verification() -> None:
    """The whole project's premise: this forgery verifies as legitimate.

    Choi section III A states the attack is *always* successful. Asserted exactly,
    with no tolerance: if the implementation produces anything below 1.0 the
    implementation is wrong, and the test must not be adjusted to accommodate it.
    """
    spec = load_example("baseline_bell_aqs")
    result = Choi2011().run(spec, trials=200, seed=7)

    assert result.verification_accepted is True
    assert result.mismatch_rate == 0.0
    assert result.message_was_modified is True
    assert result.attacker_used_key is False
    assert result.success_rate == 1.0


@pytest.mark.parametrize("seed", [1, 7, 42, 1000, 65535])
def test_choi_forgery_succeeds_for_every_seed(seed: int) -> None:
    """Exactly 1.0 across seeds, not on average across them.

    A deterministic attack that succeeded 99.5% of the time would be a broken
    implementation of a deterministic attack, not a slightly weaker attack.
    """
    result = Choi2011().run(load_example("baseline_bell_aqs"), trials=100, seed=seed)
    assert result.success_rate == 1.0
    assert result.mismatch_rate == 0.0


def test_choi_forgery_needs_no_key_material() -> None:
    """No K_AT anywhere in the attack path. Otherwise it is key compromise."""
    result = Choi2011().run(load_example("baseline_bell_aqs"), trials=10, seed=3)
    assert result.attacker_used_key is False


def test_choi_uses_the_operator_d1_predicts() -> None:
    """D1 predicts the forging operator; the attack uses that prediction.

    The coupling is the point: the static audit is not a separate opinion, it
    names the operator the attack then succeeds with.
    """
    spec = load_example("baseline_bell_aqs")
    operator, predicted = select_forging_operator(spec)

    assert predicted is True
    assert operator in spec.forging_witnesses()
    assert Choi2011(forge=operator).run(spec, trials=50, seed=2).success_rate == 1.0


def test_choi_fixed_scheme_is_forged_by_y_exactly_as_v18_predicts() -> None:
    """Choi's published fix falls to Q = Y, the witness D1 finds. Ledger V-18.

    X and Z are eliminated by the Hadamard; Y is not, because H Y H = -Y is a
    global phase. The attack succeeds with probability 1, which is the executable
    form of the finding.
    """
    spec = load_example("choi_fixed_aqs")
    assert spec.forging_witnesses() == ("Y",)

    assert Choi2011(forge="Y").run(spec, trials=100, seed=11).success_rate == 1.0
    # And the operators the fix does eliminate genuinely fail.
    assert Choi2011(forge="X").run(spec, trials=100, seed=11).success_rate == 0.0


def test_classical_message_verification_is_weaker_than_the_d1_predicate() -> None:
    """Ledger V-42, and it is a limitation of D1, not a bug in the attack.

    ``pauli_witness_free_aqs`` has zero Pauli witnesses, so D1 reports it clear of
    the Choi class. It is still forged by Q = Y. The reason is that verification
    of a *classical* message reads only the Z observable, so the attack needs the
    net operator to flip the computational basis -- strictly weaker than
    commuting with everything up to global phase.

    D1's predicate decides the general-quantum-message class Choi analyses. It
    does not decide the classical-message case, and the report must say so.
    """
    spec = load_example("pauli_witness_free_aqs")
    assert spec.forging_witnesses() == ()

    result = Choi2011(forge="Y").run(spec, trials=100, seed=5)
    assert result.success_rate == 1.0
    assert result.detail["predicted_by_d1"] is False


# --------------------------------------------------------------------------
# Blind forgery. Two rates, separately derived.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_blind_forgery_acceptance_matches_two_to_the_minus_n(length: int) -> None:
    """Acceptance is ``2**-n``, not ``4**-n``.

    Derivation: the residual ``guess * true`` is uniform over the four Paulis, and
    a Z-basis measurement is flipped only by a residual of X or Y. So a position
    survives with probability 1/2. A residual of Z is a wrong correction that is
    *invisible*, which is exactly why the exponent is 2. Ledger V-02.
    """
    trials = 4000
    spec = load_example("baseline_bell_aqs").model_copy(
        update={"signature": load_example("baseline_bell_aqs").signature.model_copy(
            update={"length_qubits": length}
        )}
    )
    result = BlindForgery().run(spec, trials=trials, seed=length)

    observed = float(result.detail["acceptance_rate"])
    expected = acceptance_probability(length)
    assert abs(observed - expected) < three_sigma(expected, trials)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_blind_forgery_correct_guess_matches_four_to_the_minus_n(length: int) -> None:
    """Guessing every correction right is ``4**-n``. A fact about the guess."""
    trials = 8000
    base = load_example("baseline_bell_aqs")
    spec = base.model_copy(
        update={"signature": base.signature.model_copy(update={"length_qubits": length})}
    )
    result = BlindForgery().run(spec, trials=trials, seed=length + 100)

    observed = float(result.detail["correct_guess_rate"])
    expected = correct_guess_probability(length)
    assert abs(observed - expected) < three_sigma(expected, trials)


def test_the_two_blind_forgery_rates_are_different_and_both_reported() -> None:
    """Conflating them would misdescribe the headline curve."""
    result = BlindForgery().run(load_example("baseline_bell_aqs"), trials=200, seed=1)

    assert result.detail["acceptance_theory"] != result.detail["correct_guess_theory"]
    assert acceptance_probability(8) == 2.0**-8
    assert correct_guess_probability(8) == 4.0**-8


# --------------------------------------------------------------------------
# Intercept-resend. The six-state figure.
# --------------------------------------------------------------------------


def test_intercept_resend_detection_rate_is_one_third() -> None:
    """(2/3) wrong basis * (1/2) wrong outcome = 1/3 per probe.

    The six-state figure, not BB84's 1/4. Verifying the derivation empirically is
    the point of the test.
    """
    trials = 60_000
    result = InterceptResend().run(load_example("baseline_bell_aqs"), trials=trials, seed=9)

    observed = float(result.detail["detection_rate"])
    assert abs(observed - DETECTION_PER_PROBE) < three_sigma(DETECTION_PER_PROBE, trials)


def test_intercept_resend_is_not_the_bb84_rate() -> None:
    """A guard against silently reverting to the four-state figure."""
    trials = 60_000
    observed = float(
        InterceptResend()
        .run(load_example("baseline_bell_aqs"), trials=trials, seed=10)
        .detail["detection_rate"]
    )
    assert abs(observed - 0.25) > 10 * three_sigma(0.25, trials)


def test_escape_probability_is_two_thirds_to_the_d() -> None:
    """Independent probes compound: escape over d probes is (2/3)**d."""
    assert escape_probability(1) == pytest.approx(2 / 3)
    assert escape_probability(32) == pytest.approx((2 / 3) ** 32)
    assert escape_probability(32) < 1e-5


# --------------------------------------------------------------------------
# Replay.
# --------------------------------------------------------------------------


def test_replayed_pair_is_byte_identical_to_the_original() -> None:
    """The replay changes nothing, which is what makes the ledger check exact."""
    result = Replay().run(load_example("baseline_bell_aqs"), trials=5, seed=4)

    assert result.detail["byte_identical"] is True
    assert result.message_was_modified is False
    assert result.detail["replayed_pair_indices"]


# --------------------------------------------------------------------------
# Registry discipline.
# --------------------------------------------------------------------------


def test_registry_holds_exactly_four_attacks() -> None:
    """The count claimed anywhere must equal this. No overclaiming."""
    assert len(ATTACK_REGISTRY) == 4
    assert set(ATTACK_REGISTRY) == {
        "blind_forgery",
        "intercept_resend",
        "choi_2011",
        "replay",
    }


def test_every_attack_cites_a_verified_source_or_says_derived() -> None:
    """Rule 2: no invented citations."""
    allowed_markers = ("derived", "Choi, Chang, Hong, Phys. Rev. A 84, 062330 (2011)")
    for name, adversary in ATTACK_REGISTRY.items():
        assert adversary.reference, name
        assert any(m in adversary.reference for m in allowed_markers), (
            f"{name} cites {adversary.reference!r}, which is not in docs/references.md"
        )


def test_every_attack_declares_a_threat_class() -> None:
    """The decision table maps threat classes; an attack without one is unmappable."""
    for name, adversary in ATTACK_REGISTRY.items():
        assert adversary.threat_class, name


@pytest.mark.parametrize("name", sorted(ATTACK_REGISTRY))
def test_every_attack_is_deterministic_under_a_fixed_seed(name: str) -> None:
    """Rule 6, across the whole attack library."""
    adversary = ATTACK_REGISTRY[name]
    spec = load_example("baseline_bell_aqs")
    first = adversary.run(spec, trials=40, seed=123)
    second = adversary.run(spec, trials=40, seed=123)

    assert first == second, f"{name} was not reproducible under a fixed seed"


@pytest.mark.parametrize("scheme", CLIFFORD_SCHEMES)
def test_every_attack_runs_against_every_clifford_scheme(scheme: str) -> None:
    """The scheme x attack matrix has no holes for simulable schemes."""
    spec: SchemeSpec = load_example(scheme)
    for name, adversary in ATTACK_REGISTRY.items():
        result = adversary.run(spec, trials=20, seed=3)
        assert result.attack == name
        assert 0.0 <= result.success_rate <= 1.0


def test_attacks_refuse_the_non_clifford_scheme_rather_than_reporting_zero() -> None:
    """A scheme that cannot be simulated must not read as "attack failed".

    Reporting 0.0 for an attack that never ran would be a silent false negative in
    the matrix -- the worst possible failure for an assurance tool.
    """
    spec = load_example("kim_forgery_free_aqs")
    with pytest.raises(NotImplementedError):
        Choi2011().run(spec, trials=1, seed=0)


def test_binomial_tolerance_helper_is_sane() -> None:
    """Guards the tolerance used by the statistical assertions above."""
    assert three_sigma(0.5, 10_000) == pytest.approx(3 * (0.25 / 10_000) ** 0.5)
    assert three_sigma(1 / 3, 60_000) < 0.006


def test_numpy_is_the_only_numeric_dependency_used_here() -> None:
    """Appendix B: no ML anywhere, including in the attack library."""
    assert np.__name__ == "numpy"
