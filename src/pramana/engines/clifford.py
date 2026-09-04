"""Clifford engine: exact stabilizer simulation via Stim.

The primary engine. Every operation in the signing path of the schemes we model is
Clifford, so the whole protocol simulates in polynomial time by the
Gottesman-Knill theorem (Aaronson & Gottesman 2004).

Rounds are batched onto one simulator rather than constructing a new one per
round, because construction dominates runtime otherwise (build plan Layer 2a).

**Determinism.** Seeding is explicit and reproducible for a fixed Stim version on
one machine; Stim's own documentation scopes it that way, so cross-platform
byte-identity is not claimed. Ledger V-12.
"""

from __future__ import annotations

import stim

from pramana.engines.base import RoundResult, RoundResults
from pramana.protocol.teleportation import (
    CORRECTION_TABLE,
    MESSAGE_QUBIT,
    PAULI_EIGENSTATES,
    RECEIVER_HALF,
    SENDER_HALF,
    prepare,
)
from pramana.spec.schema import SchemeSpec

#: Signature qubits are drawn from the six Pauli eigenstates in a fixed rotation.
#: Layer 3 replaces this with key-derived selection; here it only has to exercise
#: all three bases so an honest round is a meaningful check.
_STATE_CYCLE: tuple[str, ...] = tuple(PAULI_EIGENSTATES)


class CliffordEngine:
    """Exact stabilizer simulation of honest signing rounds."""

    name = "clifford"

    def run_rounds(self, spec: SchemeSpec, rounds: int, seed: int) -> RoundResults:
        """Run ``rounds`` honest rounds, each teleporting the signature qubits.

        An honest round here is: for each signature qubit, teleport it through a
        fresh Bell pair, apply the correction the outcome dictates, and check the
        receiver holds the original state. Verification accepts when every qubit
        recovers.

        This is the engine-level honest round. Signing and verification proper --
        rotation, encryption, the classical outcome string, the equality test --
        are Layer 3 and build on top of it.
        """
        length = spec.signature.length_qubits
        pairs_per_qubit = spec.entanglement.pairs_per_signature_qubit

        # One simulator for the whole batch; qubits are reset between uses.
        sim = stim.TableauSimulator(seed=seed)
        results: list[RoundResult] = []
        next_pair = 0

        for index in range(rounds):
            mismatches: list[int] = []
            outcomes: list[tuple[int, int]] = []
            corrections: list[str] = []
            pair_indices: list[int] = []

            for position in range(length):
                state = _STATE_CYCLE[(index + position) % len(_STATE_CYCLE)]
                _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]

                for qubit in (MESSAGE_QUBIT, SENDER_HALF, RECEIVER_HALF):
                    sim.reset(qubit)

                prepare(sim, state)
                sim.h(SENDER_HALF)
                sim.cnot(SENDER_HALF, RECEIVER_HALF)
                sim.cnot(MESSAGE_QUBIT, SENDER_HALF)
                sim.h(MESSAGE_QUBIT)
                outcome = (int(sim.measure(MESSAGE_QUBIT)), int(sim.measure(SENDER_HALF)))

                correction = CORRECTION_TABLE[outcome]
                if correction != "I":
                    getattr(sim, correction.lower())(RECEIVER_HALF)

                recovered = (
                    sim.peek_observable_expectation(stim.PauliString("__" + observable))
                    == eigenvalue
                )
                if not recovered:
                    mismatches.append(position)

                outcomes.append(outcome)
                corrections.append(correction)
                pair_indices.extend(range(next_pair, next_pair + pairs_per_qubit))
                next_pair += pairs_per_qubit

            results.append(
                RoundResult(
                    index=index,
                    accepted=not mismatches,
                    mismatch_positions=tuple(mismatches),
                    pair_indices=tuple(pair_indices),
                    outcomes=tuple(outcomes),
                    corrections=tuple(corrections),
                )
            )

        return RoundResults(engine=self.name, seed=seed, rounds=tuple(results))
