"""Shamir ``t``-of-``n`` secret sharing over ``GF(P)``.

Used to split arbitrator key material so that reconstructing probe expectations
needs ``t`` shares. That is what removes the circularity in D5: if a single
arbitrator held the reference, a malicious arbitrator would be the reference for
its own audit.

The secret is the constant term of a degree ``t-1`` polynomial. Any ``t`` shares
determine it by Lagrange interpolation; any ``t-1`` leave every value of the
constant term equally likely, because for each candidate there is exactly one
polynomial of degree ``t-1`` through the ``t-1`` shares and that candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramana.crypto.field import P, inverse


@dataclass(frozen=True)
class Share:
    """One share. ``x`` is the evaluation point, never zero."""

    x: int
    y: int


def split(secret: int, threshold: int, shares: int, seed: int) -> tuple[Share, ...]:
    """Split ``secret`` into ``shares`` shares, ``threshold`` of which reconstruct it.

    Args:
        secret: Field element to protect.
        threshold: How many shares are needed. ``t``.
        shares: How many to produce. ``n``.
        seed: Explicit, per Rule 6.

    Raises:
        ValueError: The parameters are not a usable threshold scheme.
    """
    if not 1 <= threshold <= shares:
        raise ValueError(
            f"threshold ({threshold}) must satisfy 1 <= t <= n, with n = {shares}."
        )
    if not 0 <= secret < P:
        raise ValueError(f"secret must be a field element in [0, {P}).")

    rng = np.random.default_rng(seed)
    # Coefficients a_1..a_{t-1}; a_0 is the secret.
    coefficients = [secret] + [
        int(rng.integers(0, 1 << 62)) * int(rng.integers(1, 1 << 62)) % P
        for _ in range(threshold - 1)
    ]

    out = []
    for x in range(1, shares + 1):
        y = 0
        for coefficient in reversed(coefficients):
            y = (y * x + coefficient) % P
        out.append(Share(x=x, y=y))
    return tuple(out)


def reconstruct(shares: tuple[Share, ...] | list[Share]) -> int:
    """Recover the constant term by Lagrange interpolation at ``x = 0``.

    Raises:
        ValueError: Fewer than one share, or duplicate evaluation points.
    """
    points = list(shares)
    if not points:
        raise ValueError("need at least one share to reconstruct")
    xs = [s.x for s in points]
    if len(set(xs)) != len(xs):
        raise ValueError(f"duplicate share indices {xs}; each share must be distinct")

    total = 0
    for i, share_i in enumerate(points):
        numerator, denominator = 1, 1
        for j, share_j in enumerate(points):
            if i == j:
                continue
            numerator = (numerator * (-share_j.x)) % P
            denominator = (denominator * (share_i.x - share_j.x)) % P
        total = (total + share_i.y * numerator % P * inverse(denominator)) % P
    return total
