"""Noise engine: density-matrix simulation via Qiskit Aer.

This engine exists to characterise the honest-channel baseline, not to run large
signatures. A density matrix over ``3n`` qubits is intractable well before the
signature lengths the Clifford engine handles comfortably, so each teleportation
block is simulated on its own three qubits and the rounds are assembled from the
resulting per-shot records.

Two physical parameters, both documented in ``docs/derivations.md`` section 7:

``depolarizing_p``
    Probability that a gate is followed by a depolarizing channel, which replaces
    the state with the maximally mixed state. Applied to one- and two-qubit gates
    separately, since the two-qubit error is the dominant one on real hardware.

``damping_gamma``
    Amplitude-damping probability: the chance an excited state decays to |0>
    during an idle period. Unlike depolarizing noise this is *asymmetric* -- it
    biases towards |0> rather than towards the maximally mixed state.

**On the build plan's test "with depolarizing_p = 0, results match the Clifford
engine exactly".** That test is not achievable and is not implemented as written:
Aer and Stim draw from different RNG streams, so their outcome *sequences* can
never agree. What is meaningful, and what is tested, is that the two agree on the
deterministic quantities -- accept rate exactly 1.0 and zero mismatches on a
noiseless channel. Ledger Q-2, V-32.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

from pramana.engines.base import RoundResult, RoundResults
from pramana.engines.clifford import _reject_non_clifford
from pramana.protocol.teleportation import (
    CORRECTION_TABLE,
    PAULI_EIGENSTATES,
    anticommutes,
    expected_bit,
)
from pramana.spec.schema import SchemeSpec

_STATE_CYCLE: tuple[str, ...] = tuple(PAULI_EIGENSTATES)

#: Gates the single-qubit depolarizing channel is attached to.
_ONE_QUBIT_GATES = ("h", "x", "z", "s", "sdg", "sx", "rz")
_TWO_QUBIT_GATES = ("cx",)

#: Rotates the named observable's eigenbasis onto the computational basis, so a
#: Z measurement reads it. Y needs S-dagger then H; X needs H.
_BASIS_ROTATION: dict[str, tuple[str, ...]] = {
    "Z": (),
    "X": ("h",),
    "Y": ("sdg", "h"),
}


@dataclass(frozen=True)
class _Shot:
    """One teleportation block: Bell outcome and the receiver's raw measurement."""

    outcome: tuple[int, int]
    raw_bit: int


class NoiseEngine:
    """Density-matrix simulation of honest signing rounds under channel noise."""

    name = "noise"

    def __init__(self, *, depolarizing_p: float = 0.0, damping_gamma: float = 0.0) -> None:
        """Configure the channel.

        Args:
            depolarizing_p: Per-gate depolarizing probability, in [0, 1].
            damping_gamma: Amplitude-damping probability, in [0, 1].

        Raises:
            ValueError: Either parameter is outside [0, 1]. A probability
                outside its range is a specification error, not something to
                silently clamp.
        """
        if not 0.0 <= depolarizing_p <= 1.0:
            raise ValueError(f"depolarizing_p must be in [0, 1], got {depolarizing_p}")
        if not 0.0 <= damping_gamma <= 1.0:
            raise ValueError(f"damping_gamma must be in [0, 1], got {damping_gamma}")
        self.depolarizing_p = depolarizing_p
        self.damping_gamma = damping_gamma

    def _noise_model(self) -> NoiseModel | None:
        """Build the channel, or ``None`` when both parameters are zero.

        Returning ``None`` rather than an empty model matters: an empty model
        still forces a basis translation, and the point of the zero case is to be
        the untouched ideal channel.
        """
        if self.depolarizing_p == 0.0 and self.damping_gamma == 0.0:
            return None
        model = NoiseModel()
        if self.depolarizing_p > 0.0:
            model.add_all_qubit_quantum_error(
                depolarizing_error(self.depolarizing_p, 1), list(_ONE_QUBIT_GATES)
            )
            model.add_all_qubit_quantum_error(
                depolarizing_error(self.depolarizing_p, 2), list(_TWO_QUBIT_GATES)
            )
        if self.damping_gamma > 0.0:
            model.add_all_qubit_quantum_error(
                amplitude_damping_error(self.damping_gamma), list(_ONE_QUBIT_GATES)
            )
        return model

    def _circuit(self, state: str) -> QuantumCircuit:
        """One teleportation block, measured in the state's own basis."""
        gates, observable, _eigenvalue = PAULI_EIGENSTATES[state]
        qc = QuantumCircuit(3, 3)

        for gate in gates:
            getattr(qc, "sdg" if gate == "s_dag" else gate)(0)

        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1)

        # The correction is applied classically after the run (see _accepts), so
        # the receiver's qubit is rotated into its measurement basis directly.
        for gate in _BASIS_ROTATION[observable]:
            getattr(qc, gate)(2)
        qc.measure(2, 2)
        return qc

    def _sample(self, state: str, shots: int, seed: int) -> list[_Shot]:
        """Run one block ``shots`` times and return the per-shot records."""
        simulator = AerSimulator(method="density_matrix", noise_model=self._noise_model())
        circuit = transpile(self._circuit(state), simulator)
        memory = (
            simulator.run(circuit, shots=shots, seed_simulator=seed, memory=True)
            .result()
            .get_memory()
        )
        # Qiskit returns the classical register MSB-first, i.e. "c2c1c0"; verified
        # at runtime rather than assumed (ledger V-33).
        shots_out = []
        for record in memory:
            bits = record[::-1]
            shots_out.append(_Shot(outcome=(int(bits[0]), int(bits[1])), raw_bit=int(bits[2])))
        return shots_out

    @staticmethod
    def _accepts(state: str, shot: _Shot) -> bool:
        """Whether the receiver's corrected measurement matched expectation.

        The Pauli correction is folded in classically: a correction that
        anticommutes with the measured observable flips the outcome bit. This is
        exact, and avoids needing conditional gates inside the circuit.
        """
        _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]
        correction = CORRECTION_TABLE[shot.outcome]
        corrected = shot.raw_bit ^ int(anticommutes(correction, observable))
        return corrected == expected_bit(eigenvalue)

    def run_rounds(self, spec: SchemeSpec, rounds: int, seed: int) -> RoundResults:
        """Run ``rounds`` honest rounds under the configured channel.

        Blocks are sampled per state and then assembled into rounds in the same
        state order the Clifford engine uses, so the two engines are comparable
        round for round even though their random streams are unrelated.
        """
        _reject_non_clifford(spec)
        length = spec.signature.length_qubits
        pairs_per_qubit = spec.entanglement.pairs_per_signature_qubit

        schedule = [
            _STATE_CYCLE[(index + position) % len(_STATE_CYCLE)]
            for index in range(rounds)
            for position in range(length)
        ]

        pools: dict[str, list[_Shot]] = {}
        for offset, state in enumerate(_STATE_CYCLE):
            needed = schedule.count(state)
            if needed:
                pools[state] = self._sample(state, needed, seed + offset)

        cursors = dict.fromkeys(pools, 0)
        results: list[RoundResult] = []
        next_pair = 0

        for index in range(rounds):
            mismatches: list[int] = []
            outcomes: list[tuple[int, int]] = []
            corrections: list[str] = []
            pair_indices: list[int] = []

            for position in range(length):
                state = _STATE_CYCLE[(index + position) % len(_STATE_CYCLE)]
                shot = pools[state][cursors[state]]
                cursors[state] += 1

                if not self._accepts(state, shot):
                    mismatches.append(position)
                outcomes.append(shot.outcome)
                corrections.append(CORRECTION_TABLE[shot.outcome])
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
