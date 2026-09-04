"""Kim, Lee & Lee's forgeable-message criterion for an AQS assistant unitary.

**Source:** Kim, Lee, Lee, *Forgeable quantum messages in arbitrated quantum
signature schemes*, arXiv:1708.05111 (2018), Theorem 4.

This sits **alongside** the Pauli-witness search in ``operators.py``. The two
decide different questions and must be reported separately:

===================  ==================================================================
``forging_witnesses``  Does a non-trivial operator commute up to global phase with
                       *every* encryption and rotation operator? Decides the
                       **universal-Pauli-commutant class** -- Choi's published attack.
                       Exact: bit structures, no tolerance.
``is_forgeable``       Does *any* forgeable quantum message exist for this assistant
                       unitary? Kim's Theorem 4, **necessary and sufficient**.
                       Strictly stronger: it catches schemes the witness search clears.
===================  ==================================================================

A scheme with zero Pauli witnesses may still be forgeable under Theorem 4.
``pauli_witness_free_aqs`` is exactly such a scheme, which is why it carries that
name rather than the one it had before this criterion was implemented.

**The criterion.** Write the assistant unitary, normalised to SU(2), as

    W = w0*sigma_0 + i*w1*sigma_1 - i*w2*sigma_2 + i*w3*sigma_3

with the ``w`` real, ``w0 >= 0`` and ``sum(w^2) == 1``. For distinct
``l, m, n`` in {1, 2, 3}:

    alpha_l = w0^2 + w_l^2 - 1/2
    beta_m  = w0*w_m + w_n*w_l
    gamma_n = w0*w_n - w_l*w_m

A forgeable message exists **iff** ``alpha_l * beta_m * gamma_n == 0`` for some
distinct ``l, m, n``. The set of unitaries satisfying this is written ``W_lmn``.

The criterion is invariant under ``w -> -w`` (every term is a product of two
components or a square), so the sign branch taken when normalising to SU(2) does
not matter.

**On exactness.** The Pauli-witness search is exact integer/bit arithmetic. This
one is not: it evaluates an exact algebraic criterion in floating point, because a
general assistant unitary has irrational components. The tolerance is stated in
``ZERO_TOLERANCE`` and is *not* load-bearing for any operator this project uses --
``test_kim_tolerance_is_not_load_bearing`` measures the margin between the true
zeros and the nearest non-zero term over the whole Clifford group and asserts it is
many orders of magnitude above the tolerance. That test is what lets this module
claim a reliable verdict despite being numeric.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import stim

#: Below this magnitude a Theorem 4 term counts as zero.
#:
#: Set at 1e-6, not smaller, for a specific reason: ``stim.Tableau.to_unitary_matrix``
#: returns **complex64**, so a tableau-derived matrix carries roughly 1e-7 of
#: single-precision error and a true zero does not evaluate to better than that. A
#: tighter tolerance would classify genuine zeros as non-zero and silently invert
#: the verdict.
#:
#: This is safe because the decision is nowhere near the boundary: over the whole
#: Clifford group the smallest *non-zero* Theorem 4 term is 1/2, nearly six orders of
#: magnitude above the tolerance. ``test_kim_tolerance_is_not_load_bearing``
#: asserts that margin, so the constant cannot quietly start mattering.
ZERO_TOLERANCE = 1e-6

#: Unitarity is checked at the precision the input can actually offer.
_UNITARITY_TOLERANCE = 1e-5

#: Kim et al.'s forgery-free operator, their Theorem 4 discussion.
#:
#: **NAME COLLISION -- this is not the T gate.** Kim et al. call this operator
#: ``T``. It is (i*sigma_1 - i*sigma_2 + i*sigma_3)/sqrt(3): a pi rotation about
#: the (1, -1, 1)/sqrt(3) body diagonal. A pi rotation about a body diagonal does
#: not permute the coordinate axes, so it is **non-Clifford** -- and it is a
#: different operator from the T = diag(1, exp(i*pi/4)) phase gate, which is also
#: non-Clifford but for an unrelated reason. The registry deliberately does not
#: use the bare name ``T`` for it.
W_KIM_FORGERY_FREE: np.ndarray = (
    1j * np.array([[0, 1], [1, 0]], dtype=complex)
    - 1j * np.array([[0, -1j], [1j, 0]], dtype=complex)
    + 1j * np.array([[1, 0], [0, -1]], dtype=complex)
) / np.sqrt(3)


def su2_components(unitary: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(w0, w1, w2, w3)`` for a 2x2 unitary, in Kim's convention.

    The unitary is first normalised to SU(2) by dividing out a square root of its
    determinant, then the sign is fixed so that ``w0 >= 0``.

    Raises:
        ValueError: The input is not a 2x2 unitary, or does not decompose to a
            real ``w`` in this convention.
    """
    if unitary.shape != (2, 2):
        raise ValueError(f"expected a 2x2 matrix, got shape {unitary.shape}")
    u = np.asarray(unitary, dtype=complex)
    if not np.allclose(u @ u.conj().T, np.eye(2), atol=_UNITARITY_TOLERANCE):
        raise ValueError("matrix is not unitary")

    m = u / np.sqrt(complex(np.linalg.det(u)))

    # W = w0*I + i*w1*X - i*w2*Y + i*w3*Z expands to
    #   [[w0 + i*w3,  i*w1 - w2],
    #    [i*w1 + w2,  w0 - i*w3]]
    w0 = float(m[0, 0].real)
    w3 = float(m[0, 0].imag)
    w1 = float(m[0, 1].imag)
    w2 = float(-m[0, 1].real)

    # The convention only closes if the other two entries agree; if they do not,
    # the input is not expressible with a real w and the criterion does not apply.
    expected = np.array([[w0 + 1j * w3, 1j * w1 - w2], [1j * w1 + w2, w0 - 1j * w3]])
    if not np.allclose(m, expected, atol=_UNITARITY_TOLERANCE):
        raise ValueError(
            "matrix does not decompose to a real w in Kim's convention; "
            "Theorem 4 does not apply to it as written"
        )

    w = (w0, w1, w2, w3)
    if w0 < 0:  # the criterion is invariant under w -> -w; normalise for reporting
        w = (-w0, -w1, -w2, -w3)
    return w


def theorem4_terms(
    w: tuple[float, float, float, float],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Return ``(alpha, beta, gamma)`` keyed by index 1..3, per Theorem 4."""
    w0 = w[0]
    alpha = {i: w0**2 + w[i] ** 2 - 0.5 for i in (1, 2, 3)}
    beta: dict[int, float] = {}
    gamma: dict[int, float] = {}
    for m_idx in (1, 2, 3):
        n_idx, l_idx = (i for i in (1, 2, 3) if i != m_idx)
        beta[m_idx] = w0 * w[m_idx] + w[n_idx] * w[l_idx]
    for n_idx in (1, 2, 3):
        l_idx, m_idx = (i for i in (1, 2, 3) if i != n_idx)
        gamma[n_idx] = w0 * w[n_idx] - w[l_idx] * w[m_idx]
    return alpha, beta, gamma


def is_forgeable(
    unitary: np.ndarray, *, tolerance: float = ZERO_TOLERANCE
) -> tuple[bool, tuple[int, int, int] | None]:
    """Whether a forgeable quantum message exists for this assistant unitary.

    Kim Theorem 4, necessary and sufficient.

    Returns:
        ``(forgeable, triple)``. ``triple`` is the ``(l, m, n)`` whose product
        vanishes -- the membership witness for ``W_lmn`` -- or ``None`` when no
        forgeable message exists.
    """
    alpha, beta, gamma = theorem4_terms(su2_components(unitary))
    for l_idx, m_idx, n_idx in permutations((1, 2, 3)):
        product = alpha[l_idx] * beta[m_idx] * gamma[n_idx]
        if abs(product) < tolerance:
            return True, (l_idx, m_idx, n_idx)
    return False, None


def smallest_nonzero_term(unitary: np.ndarray, *, tolerance: float = ZERO_TOLERANCE) -> float:
    """Smallest magnitude among the Theorem 4 terms that are not zero.

    Used to show the decision is not sitting on the tolerance: if this is far
    above ``tolerance`` for every operator we evaluate, the floating-point
    evaluation cannot flip the verdict.
    """
    alpha, beta, gamma = theorem4_terms(su2_components(unitary))
    magnitudes = [
        abs(v) for group in (alpha, beta, gamma) for v in group.values() if abs(v) >= tolerance
    ]
    return min(magnitudes) if magnitudes else float("inf")


def unitary_of(tableau: stim.Tableau) -> np.ndarray:
    """The 2x2 unitary a single-qubit tableau represents, up to global phase."""
    # stim returns complex64; widen so downstream arithmetic is double precision.
    # The single-precision error in the *input* is what sets ZERO_TOLERANCE.
    return np.asarray(tableau.to_unitary_matrix(endian="little"), dtype=complex)


def commutes_up_to_phase_matrix(
    a: np.ndarray, b: np.ndarray, *, tolerance: float = ZERO_TOLERANCE
) -> bool:
    """Whether ``AB = lambda BA`` for some unit scalar, decided on matrices.

    The matrix counterpart of ``operators.commutes_up_to_phase``. Needed only when
    an operator has no tableau -- that is, when the scheme declares a non-Clifford
    assistant unitary. **It is not exact**: unlike the tableau path it compares
    floating-point matrices against ``tolerance``. Every verdict computed this way
    must be reported as numerically decided, not exactly decided.
    """
    ab = a @ b
    ba = b @ a
    index = np.unravel_index(int(np.argmax(np.abs(ba))), ba.shape)
    if abs(ba[index]) < tolerance:
        return bool(np.allclose(ab, 0, atol=tolerance))
    scale = ab[index] / ba[index]
    return bool(np.allclose(ab, scale * ba, atol=tolerance) and abs(abs(scale) - 1) < tolerance)


def forging_witnesses_matrix(
    encryption: tuple[np.ndarray, ...],
    rotation: tuple[np.ndarray, ...],
    *,
    tolerance: float = ZERO_TOLERANCE,
) -> tuple[str, ...]:
    """The Pauli-witness search, run on matrices instead of tableaus.

    Same predicate as ``operators.forging_witnesses`` -- a non-trivial Q commuting
    up to global phase with every encryption and rotation operator -- but usable
    when the encryption set contains a non-Clifford operator and therefore has no
    tableau representation.

    The candidate Q still ranges over the 24 elements of C_1/U(1): a forging
    operator outside the Clifford group remains out of scope (ledger V-09).

    Returns:
        Registry names of the witnesses, sorted.
    """
    from pramana.spec.operators import clifford_group, name_of

    identity = np.eye(2, dtype=complex)
    operators = list(encryption) + list(rotation)
    found: list[str] = []
    for tableau in clifford_group():
        candidate = unitary_of(tableau)
        if commutes_up_to_phase_matrix(candidate, identity, tolerance=tolerance) and np.allclose(
            candidate
            / candidate[np.unravel_index(int(np.argmax(np.abs(candidate))), candidate.shape)],
            identity,
            atol=tolerance,
        ):
            continue  # the identity is trivial by definition
        if all(commutes_up_to_phase_matrix(candidate, op, tolerance=tolerance) for op in operators):
            found.append(name_of(tableau))
    return tuple(sorted(found))
