"""Probe-round scheduling and expectation derivation.

Probe positions and states are derived from shared key material so that an
attacker without the key cannot predict which rounds are probes or what
outcomes to expect.

**Primitive:** HMAC-SHA256, used as a PRF (FIPS 198-1). It is in Python's
standard library (``hmac`` + ``hashlib``). It is used here to generate the
probe schedule, not as a MAC -- no new reference row is needed.

**Correctness test only.** Layer 3's honest-probe test asserts probability 1
on the Clifford engine. This is a correctness test -- it verifies that the
signing and verification path preserves probe states perfectly on a noiseless
channel. It is **not** a calibration source and will **not** be passed to D2.
D2's baseline ``p_0`` comes from the noise engine, with the declared ``p_min``
floor from the scheme spec.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

from pramana.protocol.teleportation import PAULI_EIGENSTATES, expected_bit

#: The six Pauli eigenstates in a fixed order, for indexed selection.
_PROBE_STATES: tuple[str, ...] = tuple(PAULI_EIGENSTATES.keys())


def derive_probe_schedule(
    key: bytes, total_rounds: int, probe_count: int
) -> frozenset[int]:
    """Determine which round indices are probe rounds.

    Uses HMAC-SHA256 as a PRF keyed by ``key``, applied to each round index.
    The ``probe_count`` rounds with the smallest PRF outputs are chosen as
    probes. This is equivalent to a Fisher-Yates shuffle truncated at
    ``probe_count``, but uses a comparison sort on PRF values instead, which
    is simpler and deterministic without needing a stateful PRNG.

    Args:
        key: Shared key material (assumption A-1).
        total_rounds: The total number of rounds in the signing session.
        probe_count: How many of those rounds are probes.

    Returns:
        Frozenset of round indices designated as probes.

    Raises:
        ValueError: If ``probe_count > total_rounds``.
    """
    if probe_count > total_rounds:
        raise ValueError(
            f"probe_count ({probe_count}) exceeds total_rounds ({total_rounds})"
        )
    if probe_count == 0:
        return frozenset()

    # PRF: HMAC-SHA256(key, "probe_schedule" || round_index_as_4_bytes)
    prefix = b"probe_schedule"
    tagged: list[tuple[bytes, int]] = []
    for i in range(total_rounds):
        tag = hmac.new(key, prefix + struct.pack(">I", i), hashlib.sha256).digest()
        tagged.append((tag, i))

    # Sort by PRF output and take the first `probe_count`.
    tagged.sort(key=lambda t: t[0])
    return frozenset(idx for _, idx in tagged[:probe_count])


def select_probe_state(key: bytes, round_index: int) -> str:
    """Select a probe state for a given round, derived from key material.

    Returns one of the six Pauli eigenstates. The selection is deterministic
    and unpredictable without ``key``.

    Args:
        key: Shared key material.
        round_index: The round this probe belongs to.

    Returns:
        A key from ``PAULI_EIGENSTATES``, e.g. ``"|+i>"``.
    """
    tag = hmac.new(
        key, b"probe_state" + struct.pack(">I", round_index), hashlib.sha256
    ).digest()
    # Use the first 4 bytes as an unsigned int, mod 6.
    idx = struct.unpack(">I", tag[:4])[0] % len(_PROBE_STATES)
    return _PROBE_STATES[idx]


def expected_probe_outcome(state: str) -> tuple[str, int, int]:
    """The analytical expectation for a probe state.

    This is the correctness reference -- not a simulation result. On a noiseless
    Clifford channel, the probe must recover this outcome with probability 1.
    It is **not** passed to D2 as a calibration source.

    Args:
        state: A key from ``PAULI_EIGENSTATES``.

    Returns:
        A tuple ``(observable, eigenvalue, expected_bit)`` where ``observable``
        is ``"X"``, ``"Y"``, or ``"Z"``, ``eigenvalue`` is +1 or -1, and
        ``expected_bit`` is the measurement bit (0 for +1, 1 for -1).
    """
    _gates, observable, eigenvalue = PAULI_EIGENSTATES[state]
    return observable, eigenvalue, expected_bit(eigenvalue)
