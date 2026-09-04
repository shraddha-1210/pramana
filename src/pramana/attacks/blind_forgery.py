"""Blind forgery: the attacker guesses the Pauli corrections uniformly at random.

Derived, no citation needed. It is the simplest attack and it validates the attack
harness itself.

**Two different rates, reported separately.** Conflating them is how a headline
curve ends up meaning something other than it claims:

``correct_guess_rate``
    The attacker guesses *every* correction exactly right. Four corrections per
    position, uniform and independent, so the rate is ``4**-n``. This is a
    statement about the guess, not about the protocol.

``acceptance_rate``
    Verification *accepts*, which is what a forger actually needs. This is **not**
    ``4**-n``. Write the residual as ``R = guess * true``: for a uniform guess the
    residual is uniform over the four Paulis. Verification of a classical message
    measures the ``Z`` observable, and a residual flips that outcome exactly when
    it anticommutes with ``Z`` -- that is, when ``R`` is ``X`` or ``Y``. So a
    position survives with probability ``2/4 = 1/2``, and

        acceptance_rate = 2**-n

    A wrong correction is only *sometimes* detectable: a residual of ``Z`` is
    invisible to a ``Z``-basis measurement. That is why the acceptance exponent is
    ``2`` and not ``4``. Ledger V-02.

    The residual stays uniform after decryption because conjugating by the
    rotation and encryption Paulis is a bijection on the Pauli group, so the
    exponent does not depend on the scheme's factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramana.attacks.base import AttackResult, ThreatClass
from pramana.protocol.signing import sign_and_verify_round
from pramana.protocol.teleportation import PAULI_NAMES
from pramana.spec.schema import SchemeSpec


def correct_guess_probability(length: int) -> float:
    """``4**-n``: every one of ``n`` corrections guessed right."""
    return 4.0**-length


def acceptance_probability(length: int) -> float:
    """``2**-n``: verification accepts. See the module docstring for the derivation."""
    return 2.0**-length


@dataclass
class BlindForgery:
    """Uniform random guessing of the correction string."""

    name: str = "blind_forgery"
    reference: str = "derived"
    threat_class: str = ThreatClass.OUTSIDER_FORGERY

    def run(self, spec: SchemeSpec, trials: int, seed: int) -> AttackResult:
        """Guess the corrections ``trials`` times and count both rates."""
        length = spec.signature.length_qubits
        rng = np.random.default_rng(seed)
        message = tuple(int(b) for b in rng.integers(0, 2, length))

        accepted = 0
        exact = 0
        for trial in range(trials):
            honest, _ok, _mm = sign_and_verify_round(message, spec, seed=trial)
            guess = tuple(PAULI_NAMES[i] for i in rng.integers(0, 4, length))
            _sm, ok, _mismatch = sign_and_verify_round(
                message, spec, seed=trial, override_corrections=guess
            )
            accepted += int(ok)
            exact += int(guess == honest.corrections)

        return AttackResult(
            attack=self.name,
            trials=trials,
            successes=accepted,
            verification_accepted=accepted > 0,
            mismatch_rate=1.0 - accepted / trials if trials else 0.0,
            message_was_modified=False,
            attacker_used_key=False,
            detail={
                "signature_length": length,
                "correct_guess_rate": exact / trials if trials else 0.0,
                "correct_guess_theory": correct_guess_probability(length),
                "acceptance_rate": accepted / trials if trials else 0.0,
                "acceptance_theory": acceptance_probability(length),
                "note": (
                    "correct_guess_rate is 4**-n; acceptance_rate is 2**-n, because a "
                    "residual of Z is invisible to a Z-basis measurement."
                ),
            },
        )
