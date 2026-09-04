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
        is_probe: Whether this round is a probe round.
        probe_state: The Pauli eigenstate prepared for a probe round, or None.
        probe_expected: The expected measurement bit for the probe, or None.
        probe_observed: The observed measurement bit for the probe, or None.
    """

    index: int
    accepted: bool
    mismatch_positions: tuple[int, ...] = ()
    pair_indices: tuple[int, ...] = ()
    outcomes: tuple[tuple[int, int], ...] = ()
    corrections: tuple[str, ...] = ()
    is_probe: bool = False
    probe_state: str | None = None
    probe_expected: int | None = None
    probe_observed: int | None = None


@dataclass(frozen=True)
class RoundResults:
    """The result of running many rounds.

    Probe fields are populated by Layer 3. Layer 2 engine runs that do not
    interleave probes leave them at their defaults (``is_probe=False``).
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

    @property
    def probe_mismatch_rate(self) -> float | None:
        """Fraction of probe rounds whose observed bit differed from expected.

        Returns None if no probe rounds exist, so a caller cannot mistake
        "no probes" for "all probes matched".
        """
        probes = [r for r in self.rounds if r.is_probe]
        if not probes:
            return None
        mismatches = sum(1 for r in probes if r.probe_observed != r.probe_expected)
        return mismatches / len(probes)

    def fingerprint(self) -> tuple[object, ...]:
        """A hashable summary used to assert bit-identical reruns (Rule 6)."""
        return tuple(
            (
                r.index, r.accepted, r.mismatch_positions, r.pair_indices,
                r.outcomes, r.corrections,
                r.is_probe, r.probe_state, r.probe_expected, r.probe_observed,
            )
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
