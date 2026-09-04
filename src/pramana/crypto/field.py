"""The prime field shared by the MAC and the secret-sharing scheme.

``P = 2**127 - 1`` is a Mersenne prime. Its primality is **not taken on trust**:
``lucas_lehmer_is_prime`` is a complete deterministic primality proof for Mersenne
numbers, and ``test_crypto.py`` runs it. A composite modulus would silently break
both the AXU bound and Shamir's threshold property, so it is proved rather than
remembered.

Lucas-Lehmer: for an odd prime ``q``, ``M = 2**q - 1`` is prime iff ``s_{q-2} == 0``
where ``s_0 = 4`` and ``s_{i+1} = s_i**2 - 2 (mod M)``.
"""

from __future__ import annotations

#: Mersenne exponent. Prime, and small enough that Lucas-Lehmer runs instantly.
MERSENNE_EXPONENT = 127

#: The field modulus. 2**127 - 1.
P = (1 << MERSENNE_EXPONENT) - 1

#: Bits of field element, used for key-material accounting.
FIELD_BITS = MERSENNE_EXPONENT


def lucas_lehmer_is_prime(exponent: int) -> bool:
    """Whether ``2**exponent - 1`` is prime, by the Lucas-Lehmer test.

    A complete deterministic proof for Mersenne numbers, not a probabilistic one.
    """
    if exponent == 2:
        return True
    modulus = (1 << exponent) - 1
    s = 4
    for _ in range(exponent - 2):
        s = (s * s - 2) % modulus
    return s == 0


def is_prime_trial(n: int) -> bool:
    """Trial division, for checking the small exponent only."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def inverse(a: int, modulus: int = P) -> int:
    """Multiplicative inverse in the field.

    Raises:
        ZeroDivisionError: ``a`` is zero modulo the field.
    """
    a %= modulus
    if a == 0:
        raise ZeroDivisionError("zero has no multiplicative inverse")
    return pow(a, modulus - 2, modulus)
