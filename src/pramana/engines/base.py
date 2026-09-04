"""The engine interface, and the per-round record every engine produces.

Three backends sit behind this: the Clifford engine (Stim) is primary, the noise
engine (Qiskit Aer) characterises the honest-channel baseline, and both are
required to be deterministic under an explicit seed (build plan Rule 6).

**On the third engine in the build plan.** Layer 2c specified a SymPy algebra
engine whose "only consumer is detector D1". That role is already filled, exactly
and without SymPy, by ``pramana.spec.operators`` -- which represents Clifford
operators in the quotient group C_1/U(1) as bit structures and decides commutation
up to global phase by tableau equality. Building a second, weaker algebra module
would duplicate it. See ledger V-31.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pramana.spec.schema import SchemeSpec


@dataclass(frozen=True)
class RoundResult:
    """One signing round.

    Attributes:
        index: Round number, from 0.
        accepted: Whether verification accepted the round.
        mismatch_positions: Qubit positions that failed to verify. Exposed rather
            than only a boolean because the detectors need the detail (build plan
            Layer 3).
        pair_indices: Entangled-pair indices consumed. The entanglement ledger
            (D4) turns these into an exact freshness check.
        outcomes: Per-qubit Bell-measurement outcomes.
        corrections: Per-qubit Pauli correction applied by the receiver.
    """

    index: int
    accepted: bool
    mismatch_positions: tuple[int, ...] = ()
    pair_indices: tuple[int, ...] = ()
    outcomes: tuple[tuple[int, int], ...] = ()
    corrections: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundResults:
    """The result of running many rounds.

    Probe fields are deliberately absent: probe rounds and their key-derived
    expectations are Layer 3, and adding empty placeholders now would invite a
    detector to read a field that means nothing yet.
    """

    engine: str
    seed: int
    rounds: tuple[RoundResult, ...] = field(default_factory=tuple)

    @property
    def accept_rate(self) -> float:
        """Fraction of rounds that verified. 1.0 on an honest noiseless channel."""
        if not self.rounds:
            return 0.0
        return sum(1 for r in self.rounds if r.accepted) / len(self.rounds)

    def fingerprint(self) -> tuple[object, ...]:
        """A hashable summary used to assert bit-identical reruns (Rule 6)."""
        return tuple(
            (r.index, r.accepted, r.mismatch_positions, r.pair_indices, r.outcomes, r.corrections)
            for r in self.rounds
        )


class Engine(Protocol):
    """A simulation backend.

    Every entry point takes an explicit ``seed``; two runs with the same seed
    produce identical output.
    """

    name: str

    def run_rounds(self, spec: SchemeSpec, rounds: int, seed: int) -> RoundResults:
        """Run ``rounds`` honest signing rounds against ``spec``."""
        ...
