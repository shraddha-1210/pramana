"""Signing: key-controlled rotation, encryption, Bell measurement.

Takes a classical message (a sequence of bits) and a scheme spec, produces a
signature -- the sequence of Bell-measurement outcome pairs plus a consumed-pair
manifest.

**Correction table convention.** ``(0,0)->I, (0,1)->X, (1,0)->Z, (1,1)->Y``,
established empirically in Layer 2 (``docs/derivations.md`` section 6) and
confirmed by ``test_correction_table_is_uniquely_determined``. No other
convention exists in this codebase.

**Determinism.** Every entry point takes an explicit ``seed``. Two runs with
the same seed produce bit-identical output (Rule 6). Different seeds produce
different signatures -- the pair index is sufficient on its own (consumed
physical resource, monotonically advancing), and the RNG stream and probe
schedule reinforce this.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

from pramana.engines.base import RoundResult, RoundResults
from pramana.protocol.probes import (
    derive_probe_schedule,
    expected_probe_outcome,
    select_probe_state,
)
from pramana.protocol.teleportation import (
    CORRECTION_TABLE,
    MESSAGE_QUBIT,
    PAULI_EIGENSTATES,
    RECEIVER_HALF,
    SENDER_HALF,
    anticommutes,
    expected_bit,
    prepare,
)
from pramana.spec.schema import SchemeSpec


@dataclass(frozen=True)
class SignedMessage:
    """The output of signing a classical message.

    Attributes:
        message: The original message bits.
        outcomes: Per-qubit Bell-measurement outcomes (m0, m1).
        corrections: Per-qubit Pauli correction the receiver needs.
        pair_indices: Entangled-pair indices consumed. Monotonically advancing,
            never reused (V-36). Physics forbids reuse.
        seed: The seed used, so any result is reproducible.
    """

    message: tuple[int, ...]
    outcomes: tuple[tuple[int, int], ...]
    corrections: tuple[str, ...]
    pair_indices: tuple[int, ...]
    seed: int


def _reject_non_clifford(spec: SchemeSpec) -> None:
    """Refuse to sign with a non-Clifford scheme.

    The same scope boundary as the engine (V-40): a non-Clifford assistant
    unitary has no stabilizer tableau and cannot be applied in a Clifford
    simulation. Static audit (D1) is unaffected.
    """
    if not spec.is_clifford_simulable():
        raise NotImplementedError(
            f"scheme {spec.name!r} declares the non-Clifford assistant unitary "
            f"{spec.assistant_unitary!r}. Signing requires simulating the "
            "encryption path, which includes this operator. A Clifford engine "
            "cannot apply it. Static analysis (D1) still applies."
        )


def _reject_three_or_more_rotations(spec: SchemeSpec) -> None:
    """Refuse to sign with a scheme that declares 3+ rotations.

    The rotation operators for ``rotation_count >= 3`` are not modelled: sigma_x
    and sigma_z already generate the single-qubit Pauli group mod phase (Z_2 x
    Z_2, rank 2), so there is no third independent Pauli rotation on one qubit.
    What the third rotation actually is has not been derived from any source we
    have read (Q-14).
    """
    if spec.rotation_count > 2:
        raise NotImplementedError(
            f"scheme {spec.name!r} declares rotation_count={spec.rotation_count}. "
            "The rotation operators for three or more rotations are not modelled: "
            "sigma_x and sigma_z generate the single-qubit Pauli group mod phase "
            "(rank 2), so no third independent Pauli rotation exists on one qubit. "
            "What the third rotation is has not been derived from any source we "
            "have read (Q-14). Do not guess."
        )


def _apply_clifford(sim: stim.TableauSimulator, name: str, qubit: int) -> None:
    """Apply a named Clifford gate to a qubit.

    Handles the six single-qubit Clifford generators in the registry.
    """
    lower = name.lower()
    if lower == "s_dag":
        sim.s_dag(qubit)
    elif lower == "sqrt_x_dag":
        sim.sqrt_x_dag(qubit)
    else:
        getattr(sim, lower)(qubit)


def _apply_inverse_clifford(sim: stim.TableauSimulator, name: str, qubit: int) -> None:
    """Apply the inverse of a named Clifford gate.

    For Paulis and H, the inverse is the gate itself (self-inverse).
    For S, the inverse is S_dag. For S_dag, the inverse is S.
    """
    if name in ("I", "X", "Y", "Z", "H"):
        _apply_clifford(sim, name, qubit)
    elif name == "S":
        sim.s_dag(qubit)
    elif name == "S_dag":
        sim.s(qubit)
    else:
        raise NotImplementedError(
            f"inverse of {name!r} is not implemented. Extend _apply_inverse_clifford."
        )


def sign_and_verify_round(
    message_bits: tuple[int, ...],
    spec: SchemeSpec,
    seed: int,
    *,
    pair_start: int = 0,
) -> tuple[SignedMessage, bool, tuple[int, ...]]:
    """Sign a classical message and immediately verify it.

    This is the integrated honest-round operation. In the simulation, both
    parties operate on the same quantum state:

    1. Sender prepares |0> or |1>, creates Bell pair, applies rotation and
       encryption, performs Bell measurement.
    2. Receiver applies the correction from CORRECTION_TABLE, undoes encryption
       and rotation, checks the Z observable.

    For an honest round on a noiseless Clifford channel, the receiver always
    recovers the original bit. Acceptance is exactly 1.0 (V-01).

    Args:
        message_bits: Classical message as a tuple of bits (0 or 1).
        spec: The scheme specification.
        seed: Explicit seed for determinism (Rule 6).
        pair_start: Starting index for pair consumption.

    Returns:
        ``(signed_message, accepted, mismatch_positions)``

    Raises:
        NotImplementedError: If the scheme is non-Clifford or has 3+ rotations.
    """
    _reject_non_clifford(spec)
    _reject_three_or_more_rotations(spec)

    sim = stim.TableauSimulator(seed=seed)
    rng_sim = stim.TableauSimulator(seed=seed + 1)  # key-bit stream

    outcomes: list[tuple[int, int]] = []
    corrections: list[str] = []
    pair_indices: list[int] = []
    mismatches: list[int] = []
    pairs_per_qubit = spec.entanglement.pairs_per_signature_qubit
    next_pair = pair_start

    right_factor = spec.signing_encryption.right_factor
    left_factor = spec.signing_encryption.left_factor

    for pos, bit in enumerate(message_bits):
        # Reset all three qubits for this position.
        for qubit in (MESSAGE_QUBIT, SENDER_HALF, RECEIVER_HALF):
            sim.reset(qubit)

        # --- SIGNING (sender's side) ---

        # 1. Prepare the message state.
        state = "|0>" if bit == 0 else "|1>"
        prepare(sim, state)

        # 2. Create the Bell pair.
        sim.h(SENDER_HALF)
        sim.cnot(SENDER_HALF, RECEIVER_HALF)

        # 3. Key-controlled rotation: R = sigma_x^a sigma_z^b.
        rng_sim.reset(0)
        rng_sim.h(0)
        key_bit_x = int(rng_sim.measure(0))
        rng_sim.reset(0)
        rng_sim.h(0)
        key_bit_z = int(rng_sim.measure(0))

        if key_bit_x:
            sim.x(MESSAGE_QUBIT)
        if key_bit_z:
            sim.z(MESSAGE_QUBIT)

        # 4. Signing encryption: U P V with P a random Pauli.
        # Apply right factor V.
        for op_name in right_factor:
            if op_name != "I":
                _apply_clifford(sim, op_name, MESSAGE_QUBIT)

        # Apply random Pauli encryption (from key material).
        rng_sim.reset(0)
        rng_sim.h(0)
        enc_bit_x = int(rng_sim.measure(0))
        rng_sim.reset(0)
        rng_sim.h(0)
        enc_bit_z = int(rng_sim.measure(0))

        if enc_bit_x:
            sim.x(MESSAGE_QUBIT)
        if enc_bit_z:
            sim.z(MESSAGE_QUBIT)

        # Apply left factor U.
        for op_name in left_factor:
            if op_name != "I":
                _apply_clifford(sim, op_name, MESSAGE_QUBIT)

        # 5. Bell measurement.
        sim.cnot(MESSAGE_QUBIT, SENDER_HALF)
        sim.h(MESSAGE_QUBIT)
        m0 = int(sim.measure(MESSAGE_QUBIT))
        m1 = int(sim.measure(SENDER_HALF))
        outcome = (m0, m1)

        correction = CORRECTION_TABLE[outcome]
        outcomes.append(outcome)
        corrections.append(correction)

        # --- VERIFICATION (receiver's side) ---

        # 6. Apply the correction to the receiver's qubit.
        if correction != "I":
            _apply_clifford(sim, correction, RECEIVER_HALF)

        # 7. Undo the left factor U (apply U†).
        for op_name in reversed(left_factor):
            if op_name != "I":
                _apply_inverse_clifford(sim, op_name, RECEIVER_HALF)

        # 8. Undo the encryption Pauli (self-inverse).
        if enc_bit_x:
            sim.x(RECEIVER_HALF)
        if enc_bit_z:
            sim.z(RECEIVER_HALF)

        # 9. Undo the right factor V (apply V†).
        for op_name in reversed(right_factor):
            if op_name != "I":
                _apply_inverse_clifford(sim, op_name, RECEIVER_HALF)

        # 10. Undo the rotation (Paulis are self-inverse).
        if key_bit_z:
            sim.z(RECEIVER_HALF)
        if key_bit_x:
            sim.x(RECEIVER_HALF)

        # 11. Check: does the receiver's qubit hold the original state?
        got = sim.peek_observable_expectation(stim.PauliString("__Z"))
        recovered_bit = 0 if got == +1 else 1
        if recovered_bit != bit:
            mismatches.append(pos)

        # Track pair consumption.
        pair_indices.extend(range(next_pair, next_pair + pairs_per_qubit))
        next_pair += pairs_per_qubit

    signed = SignedMessage(
        message=message_bits,
        outcomes=tuple(outcomes),
        corrections=tuple(corrections),
        pair_indices=tuple(pair_indices),
        seed=seed,
    )
    return signed, len(mismatches) == 0, tuple(mismatches)


def run_protocol_rounds(
    spec: SchemeSpec,
    rounds: int,
    seed: int,
    *,
    probe_key: bytes | None = None,
) -> RoundResults:
    """Run ``rounds`` honest signing+verification rounds with interleaved probes.

    This is the Layer 3 replacement for ``CliffordEngine.run_rounds``: it adds
    signing encryption, key-controlled rotation, and probe rounds on top of the
    bare teleportation the engine provides.

    Each round signs one qubit of a message drawn from the six Pauli eigenstates
    in rotation (as the Layer 2 engine does), then verifies it.

    Probe rounds are interleaved if ``probe_key`` is provided and the spec has
    probes enabled. Probe positions are derived from ``probe_key`` via HMAC-SHA256,
    making them unpredictable to an attacker without the key.

    Args:
        spec: The scheme specification.
        rounds: Total number of rounds (signing + probe).
        seed: Explicit seed (Rule 6).
        probe_key: Shared key for probe scheduling. If None, no probes.

    Returns:
        ``RoundResults`` with probe fields populated on probe rounds.
    """
    _reject_non_clifford(spec)
    _reject_three_or_more_rotations(spec)

    # Determine probe schedule.
    probe_positions: frozenset[int] = frozenset()
    if probe_key is not None and spec.probes.enabled and spec.probes.rounds_per_signature > 0:
        probe_positions = derive_probe_schedule(
            probe_key, rounds, min(spec.probes.rounds_per_signature, rounds)
        )

    sim = stim.TableauSimulator(seed=seed)
    rng_sim = stim.TableauSimulator(seed=seed + 1)

    state_cycle = tuple(PAULI_EIGENSTATES)
    results: list[RoundResult] = []
    next_pair = 0
    pairs_per_qubit = spec.entanglement.pairs_per_signature_qubit
    right_factor = spec.signing_encryption.right_factor
    left_factor = spec.signing_encryption.left_factor

    for index in range(rounds):
        is_probe = index in probe_positions

        # Select state: probe state from key, or cycling eigenstate.
        if is_probe and probe_key is not None:
            state = select_probe_state(probe_key, index)
        else:
            state = state_cycle[index % len(state_cycle)]

        _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]

        # Reset.
        for qubit in (MESSAGE_QUBIT, SENDER_HALF, RECEIVER_HALF):
            sim.reset(qubit)

        # --- SIGNING ---
        prepare(sim, state)
        sim.h(SENDER_HALF)
        sim.cnot(SENDER_HALF, RECEIVER_HALF)

        # Rotation.
        rng_sim.reset(0)
        rng_sim.h(0)
        key_bit_x = int(rng_sim.measure(0))
        rng_sim.reset(0)
        rng_sim.h(0)
        key_bit_z = int(rng_sim.measure(0))
        if key_bit_x:
            sim.x(MESSAGE_QUBIT)
        if key_bit_z:
            sim.z(MESSAGE_QUBIT)

        # Encryption: V then random-Pauli then U.
        for op_name in right_factor:
            if op_name != "I":
                _apply_clifford(sim, op_name, MESSAGE_QUBIT)

        rng_sim.reset(0)
        rng_sim.h(0)
        enc_bit_x = int(rng_sim.measure(0))
        rng_sim.reset(0)
        rng_sim.h(0)
        enc_bit_z = int(rng_sim.measure(0))
        if enc_bit_x:
            sim.x(MESSAGE_QUBIT)
        if enc_bit_z:
            sim.z(MESSAGE_QUBIT)

        for op_name in left_factor:
            if op_name != "I":
                _apply_clifford(sim, op_name, MESSAGE_QUBIT)

        # Bell measurement.
        sim.cnot(MESSAGE_QUBIT, SENDER_HALF)
        sim.h(MESSAGE_QUBIT)
        outcome = (int(sim.measure(MESSAGE_QUBIT)), int(sim.measure(SENDER_HALF)))
        correction = CORRECTION_TABLE[outcome]

        # --- VERIFICATION ---
        if correction != "I":
            _apply_clifford(sim, correction, RECEIVER_HALF)

        # Undo U†.
        for op_name in reversed(left_factor):
            if op_name != "I":
                _apply_inverse_clifford(sim, op_name, RECEIVER_HALF)

        # Undo encryption Pauli.
        if enc_bit_x:
            sim.x(RECEIVER_HALF)
        if enc_bit_z:
            sim.z(RECEIVER_HALF)

        # Undo V†.
        for op_name in reversed(right_factor):
            if op_name != "I":
                _apply_inverse_clifford(sim, op_name, RECEIVER_HALF)

        # Undo rotation.
        if key_bit_z:
            sim.z(RECEIVER_HALF)
        if key_bit_x:
            sim.x(RECEIVER_HALF)

        # Check recovery.
        got = sim.peek_observable_expectation(stim.PauliString("__" + observable))
        recovered = got == eigenvalue

        # Probe fields.
        probe_state_val: str | None = None
        probe_expected_val: int | None = None
        probe_observed_val: int | None = None
        if is_probe:
            probe_state_val = state
            _obs, _ev, probe_expected_val = expected_probe_outcome(state)
            # For probe checking: the observed bit is derived from the expectation.
            # On a noiseless honest channel, peek gives the eigenvalue exactly.
            probe_observed_val = 0 if got == +1 else 1

        pair_indices = tuple(range(next_pair, next_pair + pairs_per_qubit))
        next_pair += pairs_per_qubit

        results.append(
            RoundResult(
                index=index,
                accepted=recovered,
                mismatch_positions=() if recovered else (0,),
                pair_indices=pair_indices,
                outcomes=(outcome,),
                corrections=(correction,),
                is_probe=is_probe,
                probe_state=probe_state_val,
                probe_expected=probe_expected_val,
                probe_observed=probe_observed_val,
            )
        )

    return RoundResults(engine="clifford_l3", seed=seed, rounds=tuple(results))
