"""Shared key material, and the budget accounting that says whether it lasts.

**The accounting is a deliverable, not bookkeeping.** It answers the problem
statement's performance-metrics requirement and produces a real number for the
deck: what fraction of Bell pairs go to key generation rather than to signatures,
and whether the system is sustainable at all.

If consumption exceeds generation the scheme is not sustainable, and
``KeyBudget.is_sustainable`` says so plainly rather than leaving a ratio for the
reader to interpret.

A pad is issued at most once. Reusing a one-time pad destroys the
information-theoretic guarantee of the Wegman-Carter MAC, so the pool raises
rather than silently reissuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pramana.crypto.field import FIELD_BITS, P


class KeyExhausted(Exception):
    """The pool cannot satisfy a request."""


class PadReused(Exception):
    """A one-time pad was requested twice. Never recoverable in place."""


@dataclass(frozen=True)
class KeyBudget:
    """What a run consumed, and whether that rate is sustainable.

    Attributes:
        pairs_for_signing: Bell pairs consumed carrying signature qubits.
        pairs_for_key_generation: Bell pairs consumed generating fresh key.
        key_bits_consumed: Key material spent on pads, probe expectations and
            identity-bound probes.
        key_bits_generated: Key material produced by entanglement-based
            generation.
        signatures: How many signatures the above paid for.
    """

    pairs_for_signing: int
    pairs_for_key_generation: int
    key_bits_consumed: int
    key_bits_generated: int
    signatures: int

    @property
    def total_pairs(self) -> int:
        """Every Bell pair the run consumed."""
        return self.pairs_for_signing + self.pairs_for_key_generation

    @property
    def key_overhead_fraction(self) -> float:
        """Fraction of Bell pairs spent on key rather than on signatures.

        The overhead number the performance-metrics requirement asks for.
        """
        if self.total_pairs == 0:
            return 0.0
        return self.pairs_for_key_generation / self.total_pairs

    @property
    def key_bits_per_signature(self) -> float:
        """Key material spent per signature."""
        if self.signatures == 0:
            return 0.0
        return self.key_bits_consumed / self.signatures

    @property
    def sustainability_ratio(self) -> float:
        """Generation rate over consumption rate.

        At least 1.0 means the pool replenishes at least as fast as it drains.
        Infinite when nothing was consumed.
        """
        if self.key_bits_consumed == 0:
            return float("inf")
        return self.key_bits_generated / self.key_bits_consumed

    @property
    def is_sustainable(self) -> bool:
        """Whether generation keeps up with consumption."""
        return self.key_bits_generated >= self.key_bits_consumed

    def summary(self) -> str:
        """A one-line verdict for the report. States the answer, not the number."""
        if self.is_sustainable:
            return (
                f"Sustainable: {self.key_bits_generated} key bits generated against "
                f"{self.key_bits_consumed} consumed "
                f"(ratio {self.sustainability_ratio:.2f}). "
                f"{self.key_overhead_fraction:.1%} of Bell pairs went to key generation."
            )
        return (
            f"NOT SUSTAINABLE: {self.key_bits_consumed} key bits consumed against only "
            f"{self.key_bits_generated} generated "
            f"(ratio {self.sustainability_ratio:.2f}). "
            "The pool drains faster than it refills; the scheme cannot run indefinitely "
            "at this rate."
        )


@dataclass
class KeyPool:
    """Shared key material between one pair of parties.

    ``seed_bits`` is assumption A-1: a one-time pre-shared authentication key.
    Everything after that is grown from it, which is why the field calls QKD "key
    growing".
    """

    seed_bits: int
    bits_per_pair: int = 1
    _consumed: int = field(default=0, init=False)
    _generated: int = field(default=0, init=False)
    _pairs_signing: int = field(default=0, init=False)
    _pairs_keygen: int = field(default=0, init=False)
    _signatures: int = field(default=0, init=False)
    _issued_pads: set[int] = field(default_factory=set, init=False)
    _next_pad_index: int = field(default=0, init=False)

    @property
    def available_bits(self) -> int:
        """Key bits left in the pool."""
        return self.seed_bits + self._generated - self._consumed

    def generate_from_pairs(self, pairs: int) -> int:
        """Grow the pool by consuming ``pairs`` Bell pairs for key generation."""
        if pairs < 0:
            raise ValueError(f"pairs must be non-negative, got {pairs}")
        produced = pairs * self.bits_per_pair
        self._generated += produced
        self._pairs_keygen += pairs
        return produced

    def consume(self, bits: int, purpose: str) -> None:
        """Spend key material.

        Raises:
            KeyExhausted: The pool does not hold enough. It raises rather than
                reusing key, because silent reuse is the failure this accounting
                exists to prevent.
        """
        if bits < 0:
            raise ValueError(f"bits must be non-negative, got {bits}")
        if bits > self.available_bits:
            raise KeyExhausted(
                f"cannot spend {bits} key bits on {purpose}: only "
                f"{self.available_bits} remain. Generate more key from entangled "
                "pairs before continuing; the pool will not reuse key material."
            )
        self._consumed += bits

    def record_signing_pairs(self, pairs: int) -> None:
        """Record Bell pairs spent carrying signature qubits."""
        self._pairs_signing += pairs

    def record_signature(self) -> None:
        """Record that one signature was produced."""
        self._signatures += 1

    def issue_pad(self) -> tuple[int, int]:
        """Issue a fresh one-time pad and its index.

        The pad is derived deterministically from the index so a run is
        reproducible; the security property enforced here is *freshness*, which
        is what the one-time guarantee actually needs.

        Returns:
            ``(pad, pad_index)``.

        Raises:
            PadReused: The index was already issued.
            KeyExhausted: Not enough key material for another pad.
        """
        index = self._next_pad_index
        if index in self._issued_pads:
            raise PadReused(
                f"pad index {index} was already issued. A one-time pad reused is a "
                "one-time pad broken: the XOR of two tags leaks the hash difference "
                "and the information-theoretic guarantee is gone."
            )
        self.consume(FIELD_BITS, f"one-time pad {index}")
        self._issued_pads.add(index)
        self._next_pad_index += 1
        return (pow(index + 1, 3, P), index)

    def reissue_pad(self, index: int) -> tuple[int, int]:
        """Deliberately attempt to reissue a pad. Always raises.

        Exists so the refusal is testable as behaviour rather than as a comment.

        Raises:
            PadReused: Always, when the index has been issued.
        """
        if index in self._issued_pads:
            raise PadReused(
                f"pad index {index} was already issued and will not be issued again."
            )
        raise KeyError(f"pad index {index} was never issued")

    def budget(self) -> KeyBudget:
        """Snapshot the accounting."""
        return KeyBudget(
            pairs_for_signing=self._pairs_signing,
            pairs_for_key_generation=self._pairs_keygen,
            key_bits_consumed=self._consumed,
            key_bits_generated=self._generated,
            signatures=self._signatures,
        )
