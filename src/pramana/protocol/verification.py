"""Verification: applies corrections from a signed message and checks recovery.

In the real protocol, the verifier holds the receiver's half of the Bell pair
and applies the correction the signer's Bell outcome dictates, then undoes the
signing operations to recover the original message. In the stabilizer
simulation, we replay the circuit and use ``peek_observable_expectation`` for
an exact check.

**Correction table convention.** ``(0,0)->I, (0,1)->X, (1,0)->Z, (1,1)->Y``,
the same table signing uses. Established empirically in Layer 2
(``docs/derivations.md`` section 6).

**Honest-probe test.** Verification of a probe round asserts probability 1 on
the Clifford engine. This is a correctness test only and not a calibration
source. It will not be passed to D2.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

from pramana.protocol.teleportation import (
    CORRECTION_TABLE,
    MESSAGE_QUBIT,
    RECEIVER_HALF,
    SENDER_HALF,
    anticommutes,
    expected_bit,
    prepare,
)
from pramana.spec.schema import SchemeSpec


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of verifying a signed message.

    Attributes:
        accepted: Whether verification accepted the message as authentic.
        mismatch_positions: Qubit positions where the recovered bit differed
            from the claimed message. Exposed rather than only ``accepted``
            because detectors need the detail.
        mismatch_rate: Fraction of qubits that mismatched.
        recovered_message: The message bits as recovered by the verifier.
    """

    accepted: bool
    mismatch_positions: tuple[int, ...]
    mismatch_rate: float
    recovered_message: tuple[int, ...]


def verify_round(
    message_bit: int,
    outcome: tuple[int, int],
    correction: str,
    sim: stim.TableauSimulator,
) -> tuple[bool, int]:
    """Verify a single qubit position by checking stabilizer expectation.

    The verifier holds the receiver's Bell half (qubit 2). After the signer's
    Bell measurement, the receiver applies the correction from
    ``CORRECTION_TABLE`` to recover the original state.

    For an honest round with a classical message (Z-basis eigenstate), the
    receiver's qubit after correction should hold the original |0> or |1>.
    This is checked via the Z observable.

    Args:
        message_bit: The claimed message bit (0 or 1).
        outcome: The Bell-measurement outcome from signing.
        correction: The Pauli correction name from ``CORRECTION_TABLE``.
        sim: The simulator, already holding the post-teleportation state on
            the receiver's qubit.

    Returns:
        ``(match, recovered_bit)`` where ``match`` is True iff the recovered
        bit equals the claimed message bit.
    """
    # Apply the correction to the receiver's qubit.
    if correction != "I":
        _apply_pauli(sim, correction, RECEIVER_HALF)

    # Check the Z observable on the receiver's qubit.
    got = sim.peek_observable_expectation(stim.PauliString("__Z"))
    recovered_bit = 0 if got == +1 else 1

    return recovered_bit == message_bit, recovered_bit


def _apply_pauli(sim: stim.TableauSimulator, name: str, qubit: int) -> None:
    """Apply a named Pauli gate. I is a no-op."""
    if name == "I":
        return
    getattr(sim, name.lower())(qubit)
