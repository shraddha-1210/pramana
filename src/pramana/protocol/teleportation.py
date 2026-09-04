"""Teleportation, and the Bell-outcome to Pauli-correction table it depends on.

**The correction table in this module was not written from memory.** The exact
bit-to-operator mapping depends on the Bell-basis ordering convention and differs
between textbooks, so build plan Layer 2 requires it be established empirically:
implement teleportation with a *parameterised* table, teleport all six Pauli
eigenstates, and keep only the assignment that recovers every one of them.

``determine_correction_table`` is that procedure, kept in the codebase rather than
run once and discarded, so ``test_correction_table_is_uniquely_determined`` can
re-derive the constant from scratch and prove it. If the convention here is ever
wrong, that test fails; it must never be adjusted to match the constant.

The circuit, which *is* safe to write directly (build plan Layer 2, "what you
already know"):

* qubit 0 holds the message; qubits 1 and 2 are a Bell pair, 1 with the sender and
  2 with the receiver;
* the sender applies CNOT(0 -> 1), then H(0), then measures both, giving two bits;
* the receiver applies the Pauli named by ``CORRECTION_TABLE`` to qubit 2.

Recorded in ``docs/derivations.md`` section 6, with the test that establishes it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import stim

#: Message qubit, sender's Bell half, receiver's Bell half.
MESSAGE_QUBIT = 0
SENDER_HALF = 1
RECEIVER_HALF = 2

#: Bell-measurement outcome ``(m0, m1)`` -> Pauli correction the receiver applies.
#:
#: **Established empirically, not quoted.** Determined by
#: ``determine_correction_table`` against all six Pauli eigenstates and all four
#: outcomes; exactly one correction per outcome recovers every state, so the table
#: is unique. See ``docs/derivations.md`` section 6 and ledger V-16.
#:
#: The ``(1, 1)`` entry is Y rather than the "XZ" many texts write. They are the
#: same correction: XZ = -iY, and the difference is a global phase, which is
#: unobservable. Y is written here because it is what the determination returns
#: and because the codebase represents corrections as single Pauli names.
CORRECTION_TABLE: dict[tuple[int, int], str] = {
    (0, 0): "I",
    (0, 1): "X",
    (1, 0): "Z",
    (1, 1): "Y",
}

#: The six Pauli eigenstates: preparation gates from |0>, the observable they are
#: an eigenstate of, and the eigenvalue. Three mutually unbiased bases, Z, X, Y.
PAULI_EIGENSTATES: dict[str, tuple[tuple[str, ...], str, int]] = {
    "|0>": ((), "Z", +1),
    "|1>": (("x",), "Z", -1),
    "|+>": (("h",), "X", +1),
    "|->": (("x", "h"), "X", -1),
    "|+i>": (("h", "s"), "Y", +1),
    "|-i>": (("h", "s_dag"), "Y", -1),
}

PAULI_NAMES: tuple[str, ...] = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class TeleportationShot:
    """One teleportation run."""

    state: str
    outcome: tuple[int, int]
    correction: str
    recovered: bool


def anticommutes(correction: str, observable: str) -> bool:
    """Whether a Pauli correction flips the sign of a Pauli observable.

    Two single-qubit Paulis anticommute exactly when both are non-identity and
    they differ. Used by engines that measure rather than inspect the state, to
    fold the correction into the expected classical bit.
    """
    if correction == "I" or observable == "I":
        return False
    return correction != observable


def expected_bit(eigenvalue: int) -> int:
    """Measurement bit for an eigenvalue: +1 -> 0, -1 -> 1."""
    return 0 if eigenvalue == +1 else 1


def _apply_pauli(sim: stim.TableauSimulator, name: str, qubit: int) -> None:
    """Apply a named single-qubit Pauli. ``I`` is a no-op."""
    if name == "I":
        return
    getattr(sim, name.lower())(qubit)


def prepare(sim: stim.TableauSimulator, state: str, qubit: int = MESSAGE_QUBIT) -> None:
    """Prepare one of the six Pauli eigenstates on ``qubit``, starting from |0>."""
    gates, _observable, _eigenvalue = PAULI_EIGENSTATES[state]
    for gate in gates:
        getattr(sim, gate)(qubit)


def bell_measure(sim: stim.TableauSimulator) -> tuple[int, int]:
    """Create the Bell pair, run the sender's Bell measurement, return ``(m0, m1)``.

    Leaves the simulator holding the receiver's half, uncorrected.
    """
    sim.h(SENDER_HALF)
    sim.cnot(SENDER_HALF, RECEIVER_HALF)
    sim.cnot(MESSAGE_QUBIT, SENDER_HALF)
    sim.h(MESSAGE_QUBIT)
    return int(sim.measure(MESSAGE_QUBIT)), int(sim.measure(SENDER_HALF))


def teleport(
    state: str,
    seed: int,
    *,
    table: dict[tuple[int, int], str] | None = None,
) -> TeleportationShot:
    """Teleport one Pauli eigenstate and report whether it was recovered exactly.

    Recovery is checked with ``peek_observable_expectation``, which reads the
    stabilizer exactly rather than sampling, so ``recovered`` is a statement about
    the state and not about a finite number of shots.

    Args:
        state: One of the keys of ``PAULI_EIGENSTATES``.
        seed: Explicit, per build plan Rule 6.
        table: Correction table to use. Defaults to ``CORRECTION_TABLE``; the
            parameter exists so tests can drive alternative tables and show that
            the recorded one is the only one that works.
    """
    corrections = CORRECTION_TABLE if table is None else table
    _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]

    sim = stim.TableauSimulator(seed=seed)
    prepare(sim, state)
    outcome = bell_measure(sim)

    correction = corrections[outcome]
    _apply_pauli(sim, correction, RECEIVER_HALF)

    got = sim.peek_observable_expectation(stim.PauliString("__" + observable))
    return TeleportationShot(
        state=state, outcome=outcome, correction=correction, recovered=got == eigenvalue
    )


def determine_correction_table(*, shots_per_state: int = 400) -> dict[tuple[int, int], list[str]]:
    """Find, for each Bell outcome, every correction that recovers all six states.

    This is the empirical procedure that establishes ``CORRECTION_TABLE`` (build
    plan Layer 2). It assumes nothing about the convention: for each outcome it
    tries all four Paulis and keeps those that work for every eigenstate on every
    shot.

    Returns:
        Outcome -> the corrections that recovered every eigenstate. A correct,
        unambiguous convention gives exactly one per outcome.
    """
    works: dict[tuple[int, int], dict[str, bool]] = defaultdict(
        lambda: dict.fromkeys(PAULI_NAMES, True)
    )
    for state in PAULI_EIGENSTATES:
        _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]
        pauli_string = stim.PauliString("__" + observable)
        for seed in range(shots_per_state):
            sim = stim.TableauSimulator(seed=seed)
            prepare(sim, state)
            outcome = bell_measure(sim)
            for candidate in PAULI_NAMES:
                _apply_pauli(sim, candidate, RECEIVER_HALF)
                got = sim.peek_observable_expectation(pauli_string)
                _apply_pauli(sim, candidate, RECEIVER_HALF)  # Paulis self-inverse
                if got != eigenvalue:
                    works[outcome][candidate] = False

    return {
        outcome: [c for c, ok in candidates.items() if ok]
        for outcome, candidates in sorted(works.items())
    }
