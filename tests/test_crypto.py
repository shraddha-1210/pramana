"""Layer 4: the classical cryptographic foundations.

Every bound here is derived, not measured. The AXU epsilon comes from the
root-count of a non-zero polynomial over a field; the threshold properties come
from Lagrange interpolation. Where a guarantee stops holding, there is a
deliberate expected-failure test saying so.
"""

from __future__ import annotations

import pytest

from pramana.crypto.arbitrator import ArbitrationPanel
from pramana.crypto.field import (
    MERSENNE_EXPONENT,
    P,
    inverse,
    is_prime_trial,
    lucas_lehmer_is_prime,
)
from pramana.crypto.key_pool import KeyExhausted, KeyPool, PadReused
from pramana.crypto.shamir import Share, reconstruct, split
from pramana.crypto.wegman_carter import (
    BLOCK_BYTES,
    AuthenticationError,
    Tag,
    axu_hash,
    make_tag,
    max_collision_probability,
    require_tag,
    verify_tag,
)

SECRET = 0xDEADBEEFCAFE1234
KEY = 0x5EED1234ABCD


# --------------------------------------------------------------------------
# The field. Proved, not assumed.
# --------------------------------------------------------------------------


def test_field_modulus_is_prime_by_lucas_lehmer() -> None:
    """P = 2**127 - 1 is prime, proved deterministically.

    Not a probabilistic test and not a remembered fact. A composite modulus would
    silently break both the AXU bound (which needs a field's root-count property)
    and Shamir's threshold guarantee (which needs invertibility).
    """
    assert is_prime_trial(MERSENNE_EXPONENT), "the Mersenne exponent must itself be prime"
    assert lucas_lehmer_is_prime(MERSENNE_EXPONENT)
    assert P == (1 << 127) - 1


def test_lucas_lehmer_rejects_a_composite_mersenne_number() -> None:
    """The primality test has power: 2**11 - 1 = 23 * 89 is composite."""
    assert is_prime_trial(11)
    assert lucas_lehmer_is_prime(11) is False


def test_inverse_is_a_multiplicative_inverse() -> None:
    for a in (1, 2, 3, SECRET, P - 1):
        assert a * inverse(a) % P == 1
    with pytest.raises(ZeroDivisionError):
        inverse(0)


# --------------------------------------------------------------------------
# Wegman-Carter.
# --------------------------------------------------------------------------


def test_a_valid_tag_verifies() -> None:
    message = b"transfer 100000 to account 42"
    tag = make_tag(KEY, pad=12345, pad_index=0, message=message)
    assert verify_tag(KEY, 12345, message, tag) is True


@pytest.mark.parametrize("position", [0, 7, 20])
def test_any_single_bit_change_fails_verification(position: int) -> None:
    """Integrity, which is exactly what the one-time pad on M_A lacks (V-07)."""
    message = bytearray(b"transfer 100000 to account 42")
    tag = make_tag(KEY, pad=999, pad_index=0, message=bytes(message))

    message[position // 8] ^= 1 << (position % 8)
    assert verify_tag(KEY, 999, bytes(message), tag) is False
    with pytest.raises(AuthenticationError, match="does not authenticate"):
        require_tag(KEY, 999, bytes(message), tag)


def test_a_wrong_key_fails_verification() -> None:
    message = b"payload"
    tag = make_tag(KEY, pad=1, pad_index=0, message=message)
    assert verify_tag(KEY + 1, 1, message, tag) is False


def test_a_wrong_pad_fails_verification() -> None:
    message = b"payload"
    tag = make_tag(KEY, pad=1, pad_index=0, message=message)
    assert verify_tag(KEY, 2, message, tag) is False


def test_length_extension_does_not_collide() -> None:
    """The length prefix makes the block encoding injective.

    Without it, b"a" and b"a\\x00" could share a block sequence and collide for a
    reason unrelated to the field, silently voiding the AXU bound.
    """
    assert axu_hash(KEY, b"a") != axu_hash(KEY, b"a\x00")
    assert axu_hash(KEY, b"") != axu_hash(KEY, b"\x00")


def test_epsilon_bound_matches_the_derivation() -> None:
    """epsilon = L / P, with L the number of field blocks.

    Derivation: H_k(m) - H_k(m') is a non-zero polynomial of degree <= L in k, so
    it has at most L roots among the P keys. See wegman_carter.py.
    """
    message = b"x" * (BLOCK_BYTES * 4)  # 4 blocks of payload, plus the length prefix
    blocks = -(-(len(message) + 8) // BLOCK_BYTES)

    assert max_collision_probability(message) == pytest.approx(blocks / P)
    # And it is negligible: far below any operational threshold.
    assert max_collision_probability(message) < 1e-30


def test_tag_is_deterministic_under_fixed_key_and_pad() -> None:
    message = b"reproducible"
    assert make_tag(KEY, 5, 0, message) == make_tag(KEY, 5, 0, message)


def test_tag_carries_its_pad_index() -> None:
    """The index is part of the tag so a verifier can refuse a reused pad."""
    tag = make_tag(KEY, 7, 3, b"m")
    assert isinstance(tag, Tag)
    assert tag.pad_index == 3


# --------------------------------------------------------------------------
# Key pool and budget.
# --------------------------------------------------------------------------


def test_pad_is_never_issued_twice() -> None:
    """A reused one-time pad is a broken one-time pad. The pool refuses."""
    pool = KeyPool(seed_bits=10_000)
    _pad, index = pool.issue_pad()
    with pytest.raises(PadReused, match="already issued"):
        pool.reissue_pad(index)


def test_successive_pads_are_distinct() -> None:
    pool = KeyPool(seed_bits=10_000)
    pads = [pool.issue_pad()[0] for _ in range(16)]
    assert len(set(pads)) == 16


def test_consumption_accounting_is_exact() -> None:
    """Available = seed + generated - consumed, with no slippage."""
    pool = KeyPool(seed_bits=1000, bits_per_pair=1)
    pool.generate_from_pairs(500)
    pool.consume(300, "probe expectations")

    assert pool.available_bits == 1000 + 500 - 300
    budget = pool.budget()
    assert budget.key_bits_generated == 500
    assert budget.key_bits_consumed == 300
    assert budget.pairs_for_key_generation == 500


def test_exhaustion_raises_rather_than_reusing_key() -> None:
    pool = KeyPool(seed_bits=100)
    with pytest.raises(KeyExhausted, match="will not reuse key material"):
        pool.consume(101, "an over-large request")


def test_budget_reports_unsustainable_plainly() -> None:
    """Consumption above generation is stated, not left as a ratio to interpret."""
    pool = KeyPool(seed_bits=10_000, bits_per_pair=1)
    pool.generate_from_pairs(100)
    pool.consume(500, "pads")

    budget = pool.budget()
    assert budget.is_sustainable is False
    assert budget.sustainability_ratio == pytest.approx(100 / 500)
    assert "NOT SUSTAINABLE" in budget.summary()


def test_budget_reports_sustainable_when_generation_keeps_up() -> None:
    pool = KeyPool(seed_bits=10_000, bits_per_pair=2)
    pool.generate_from_pairs(500)  # 1000 bits
    pool.consume(400, "pads")

    budget = pool.budget()
    assert budget.is_sustainable is True
    assert budget.sustainability_ratio == pytest.approx(1000 / 400)
    assert "Sustainable" in budget.summary()


def test_key_overhead_fraction_is_the_deck_number() -> None:
    """Fraction of Bell pairs spent on key rather than signatures."""
    pool = KeyPool(seed_bits=10_000)
    pool.record_signing_pairs(800)
    pool.generate_from_pairs(200)

    budget = pool.budget()
    assert budget.total_pairs == 1000
    assert budget.key_overhead_fraction == pytest.approx(0.2)


def test_key_bits_per_signature() -> None:
    pool = KeyPool(seed_bits=10_000)
    for _ in range(4):
        pool.record_signature()
    pool.consume(128, "pads")
    assert pool.budget().key_bits_per_signature == pytest.approx(32.0)


# --------------------------------------------------------------------------
# Shamir.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("threshold", "count"), [(2, 2), (2, 3), (3, 5), (1, 1)])
def test_threshold_shares_reconstruct_exactly(threshold: int, count: int) -> None:
    """Any t shares recover the secret exactly. Lagrange interpolation at x = 0."""
    shares = split(SECRET, threshold, count, seed=1)
    assert len(shares) == count

    from itertools import combinations

    for quorum in combinations(shares, threshold):
        assert reconstruct(list(quorum)) == SECRET


def test_fewer_than_threshold_shares_do_not_reveal_the_secret() -> None:
    """t-1 shares leave the secret undetermined.

    Interpolating t-1 points gives *a* polynomial, but not the right one: for
    every candidate constant term there is exactly one degree t-1 polynomial
    through those points, so the shares carry no information about which.
    """
    shares = split(SECRET, threshold=3, shares=5, seed=2)
    assert reconstruct(list(shares[:2])) != SECRET


def test_duplicate_share_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate share"):
        reconstruct([Share(x=1, y=5), Share(x=1, y=7)])


def test_invalid_threshold_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 <= t <= n"):
        split(SECRET, threshold=4, shares=3, seed=1)


def test_split_is_deterministic_under_a_seed() -> None:
    assert split(SECRET, 2, 3, seed=7) == split(SECRET, 2, 3, seed=7)
    assert split(SECRET, 2, 3, seed=7) != split(SECRET, 2, 3, seed=8)


# --------------------------------------------------------------------------
# Threshold arbitration. The three rows of the capability table.
# --------------------------------------------------------------------------


def test_honest_panel_agrees() -> None:
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=3, seed=3)
    verdict = panel.adjudicate(expected=SECRET)

    assert verdict.agreed is True
    assert verdict.secret == SECRET
    assert verdict.suspected == ()
    assert verdict.explanation


def test_n2_t2_cannot_even_detect_from_shares_alone() -> None:
    """n = 2, t = 2: redundancy 0. **Corrects the build plan.**

    The plan claims this configuration detects a lie. It cannot: with one quorum
    there is nothing to compare against, so a corrupted share simply yields a
    different secret with no signal. Detection needs n >= t + 1.
    """
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=2, seed=4, dishonest=(1,))

    assert panel.redundancy == 0
    assert panel.can_detect is False
    assert panel.can_identify is False

    # From shares alone the panel is unanimous, and unanimously wrong.
    blind = panel.adjudicate()
    assert blind.agreed is True
    assert blind.secret != SECRET

    # Supplying the true value catches it, but that is a test instrument, not a
    # security mechanism: a caller who knows the secret does not need the panel.
    informed = panel.adjudicate(expected=SECRET)
    assert informed.agreed is False
    assert informed.suspected == ()


def test_n3_t2_detects_a_liar_but_cannot_identify_one() -> None:
    """n = 3, t = 2: redundancy 1. Detection only. **Corrects the build plan.**

    The plan claims this identifies the liar by majority. It does not. Shamir
    shares are a Reed-Solomon code of distance d = n - t + 1 = 2, which detects
    one error and corrects none: three points that should be collinear are not,
    and each of the three pairs defines a different line, so nothing marks the
    outlier. Identification needs 2e <= n - t, i.e. n >= t + 2.
    """
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=3, seed=5, dishonest=(2,))

    assert panel.redundancy == 1
    assert panel.can_detect is True
    assert panel.can_identify is False

    verdict = panel.adjudicate()
    assert verdict.agreed is False, "a lie must at least be noticed"
    assert verdict.suspected == (), "the panel must not guess a culprit it cannot locate"
    assert "cannot be named" in verdict.explanation


@pytest.mark.parametrize("liar", [0, 1, 2, 3])
def test_n4_t2_identifies_whichever_arbitrator_lies(liar: int) -> None:
    """n = 4, t = 2: redundancy 2, the smallest panel that identifies one liar.

    The three quorums avoiding the liar all reconstruct the true secret, giving a
    genuine majority; the three containing it disagree with each other.
    """
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=4, seed=6, dishonest=(liar,))

    assert panel.can_identify is True
    verdict = panel.adjudicate()

    assert verdict.agreed is False
    assert verdict.suspected == (liar,)
    assert verdict.secret == SECRET, "the majority value must be the true secret"


def test_n5_t3_identifies_one_liar() -> None:
    """Redundancy 2 again, at a larger threshold."""
    panel = ArbitrationPanel.build(SECRET, threshold=3, count=5, seed=11, dishonest=(3,))
    verdict = panel.adjudicate()

    assert panel.can_identify is True
    assert verdict.suspected == (3,)
    assert verdict.secret == SECRET


@pytest.mark.xfail(
    reason=(
        "SECURITY BOUNDARY, assumption A-2. With t = 2 of n = 4 arbitrators "
        "colluding, the quorum of the two liars is self-consistent and the panel "
        "cannot correct two errors (2e = 4 > n - t = 2). This documents where the "
        "guarantee stops rather than pretending it does not. Expected to fail."
    ),
    strict=True,
)
def test_two_colluding_arbitrators_defeat_identification() -> None:
    """Two colluding arbitrators break identification. Expected failure."""
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=4, seed=7, dishonest=(2, 3))

    # The assertion that WOULD hold if the scheme were secure against t colluding
    # arbitrators. It does not hold, and must not be weakened.
    assert panel.adjudicate().suspected == (2, 3)


def test_collusion_is_still_detected_even_when_not_identified() -> None:
    """The honest fallback: disagreement stays visible even when unattributable."""
    panel = ArbitrationPanel.build(SECRET, threshold=2, count=4, seed=8, dishonest=(2, 3))
    assert panel.adjudicate().agreed is False


def test_capability_boundaries_follow_the_reed_solomon_bound() -> None:
    """detect iff n - t >= 1; identify iff n - t >= 2. Derived, not measured."""
    for threshold, count in [(2, 2), (2, 3), (2, 4), (3, 5), (3, 3), (1, 3)]:
        panel = ArbitrationPanel.build(SECRET, threshold, count, seed=1)
        redundancy = count - threshold
        assert panel.can_detect is (redundancy >= 1)
        assert panel.can_identify is (redundancy >= 2)


def test_verdicts_always_carry_an_explanation() -> None:
    """A certification tool justifies every ruling. Not optional."""
    for dishonest in ((), (0,), (0, 1)):
        panel = ArbitrationPanel.build(SECRET, 2, 3, seed=9, dishonest=dishonest)
        assert panel.adjudicate(expected=SECRET).explanation.strip()
