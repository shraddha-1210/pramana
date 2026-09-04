"""Wegman-Carter authentication of the classical channel.

**Why it is here.** The correction bits and the Bell-measurement string ``M_A``
travel on a classical channel. Choi section III A identifies OTP-protected ``M_A``
as the enabler of attack variant 2: a one-time pad gives confidentiality and *no
integrity*, so ``M_A`` is bit-flip malleable (ledger V-07). An
information-theoretically secure MAC closes that, and closes it without trading
anything away -- see ``docs/derivations.md`` section 5.5.

**Construction.** ``tag = H_k(message) XOR pad``, with ``H_k`` drawn from an
epsilon-almost-XOR-universal family and ``pad`` fresh per message
(Wegman & Carter 1981).

``H_k`` is polynomial evaluation over ``GF(P)``, ``P = 2**127 - 1``: the message is
split into field-element blocks ``m_1..m_L`` and

    H_k(m) = sum_i  m_i * k**i   (mod P)

**The epsilon bound, derived rather than quoted.** For distinct messages ``m`` and
``m'`` and any target difference ``c``,

    H_k(m) - H_k(m') = sum_i (m_i - m'_i) k**i

is a *non-zero* polynomial in ``k`` of degree at most ``L``, because the messages
differ in at least one block. A non-zero polynomial of degree ``L`` over a field
has at most ``L`` roots, so at most ``L`` of the ``P`` possible keys satisfy the
equation:

    Pr_k[ H_k(m) - H_k(m') = c ]  <=  L / P

That is the AXU property with ``epsilon = L / P``. The derivation needs only that
``GF(P)`` is a field, which is why ``P``'s primality is proved by Lucas-Lehmer
rather than assumed (``crypto/field.py``).

**The key-consumption property that makes this practical.** The hash key ``k`` may
be reused across messages; only the ``pad`` must be fresh. Key consumption is
therefore *tag-length bits per message*, not message-length bits, which is why the
key pool does not exhaust. Stated in the deck, accounted for in ``key_pool.py``.

Reusing a pad destroys the guarantee, so ``KeyPool`` refuses to issue one twice.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.crypto.field import FIELD_BITS, P

#: Bytes per message block. 15 bytes < 127 bits guarantees every block is a
#: canonical field element, so distinct block sequences stay distinct in GF(P).
BLOCK_BYTES = 15

#: Tag width in bits. One field element.
TAG_BITS = FIELD_BITS


class AuthenticationError(Exception):
    """A tag did not verify."""


def _blocks(message: bytes) -> list[int]:
    """Split a message into field elements, length-prefixed.

    The length prefix makes the encoding injective: without it ``b"a"`` and
    ``b"a\\x00"`` could share a block sequence and collide trivially, which would
    break the AXU bound for a reason that has nothing to do with the field.
    """
    framed = len(message).to_bytes(8, "big") + message
    return [
        int.from_bytes(framed[i : i + BLOCK_BYTES], "big")
        for i in range(0, len(framed), BLOCK_BYTES)
    ]


def axu_hash(key: int, message: bytes) -> int:
    """Evaluate ``H_k(message) = sum_i m_i * k**i`` over ``GF(P)``.

    Horner's rule, arranged so the polynomial has **no constant term**. That
    detail is load-bearing: with the blocks as coefficients of ``k**0..k**(L-1)``
    a single-block message would evaluate to ``m_1`` regardless of the key, so the
    tag would not depend on the key at all. Every block must multiply a positive
    power of ``k``.
    """
    accumulator = 0
    for block in _blocks(message):
        accumulator = ((accumulator + block) * key) % P
    return accumulator


def max_collision_probability(message: bytes) -> float:
    """The AXU bound ``epsilon = L / P`` for a message of this length.

    Reported rather than hard-coded so the security parameter can be shown in the
    assurance report alongside the number it applies to.
    """
    return len(_blocks(message)) / P


@dataclass(frozen=True)
class Tag:
    """An authentication tag and the pad index that must never be reused."""

    value: int
    pad_index: int


def make_tag(key: int, pad: int, pad_index: int, message: bytes) -> Tag:
    """Compute ``tag = H_k(message) XOR pad``."""
    return Tag(value=axu_hash(key, message) ^ (pad % P), pad_index=pad_index)


def verify_tag(key: int, pad: int, message: bytes, tag: Tag) -> bool:
    """Whether ``tag`` authenticates ``message`` under ``key`` and ``pad``."""
    return make_tag(key, pad, tag.pad_index, message).value == tag.value


def require_tag(key: int, pad: int, message: bytes, tag: Tag) -> None:
    """Verify, or raise.

    Raises:
        AuthenticationError: The tag does not authenticate the message.
    """
    if not verify_tag(key, pad, message, tag):
        raise AuthenticationError(
            f"tag {tag.value:#x} does not authenticate a {len(message)}-byte message "
            f"under pad index {tag.pad_index}. The message or the tag was altered in "
            "transit; reject the round rather than retrying."
        )
