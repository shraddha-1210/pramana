"""Intercept-resend on probe rounds. Standard technique, derived rate.

The attacker intercepts a probe, measures it in a randomly chosen basis, and
resends whatever it found.

**Derivation (docs/derivations.md section 3).** Probes are the six Pauli
eigenstates, spanning three mutually unbiased bases Z, X and Y.

1. The attacker picks the wrong basis with probability ``2/3``.
2. Given a wrong basis, the resent state is mutually unbiased with respect to the
   original one, so the legitimate party's outcome is uniformly random and wrong
   with probability ``1/2``.

Per-probe detection probability = ``(2/3) * (1/2) = 1/3``.

Escape probability over ``d`` probes = ``(2/3)**d``.

This is the **six-state** figure, not BB84's ``1/4``. The difference is the third
basis: with only two bases the attacker guesses right half the time instead of a
third of the time. The test verifies the derivation empirically -- that
verification is the point, not the number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from pramana.attacks.base import AttackResult, ThreatClass
from pramana.spec.schema import SchemeSpec

#: The three mutually unbiased bases the six-state probe set spans.
BASES = ("Z", "X", "Y")

#: Per-probe detection probability, derived above.
DETECTION_PER_PROBE = 1.0 / 3.0


def escape_probability(probes: int) -> float:
    """``(2/3)**d``: the attacker survives ``d`` probes undetected."""
    return (1.0 - DETECTION_PER_PROBE) ** probes


def _prepare(sim: stim.TableauSimulator, basis: str, eigenvalue: int, qubit: int) -> None:
    """Prepare the named basis eigenstate on ``qubit``, from |0>."""
    if basis == "Z":
        if eigenvalue < 0:
            sim.x(qubit)
    elif basis == "X":
        if eigenvalue < 0:
            sim.x(qubit)
        sim.h(qubit)
    else:  # Y
        if eigenvalue < 0:
            sim.x(qubit)
        sim.h(qubit)
        sim.s(qubit)


def _measure(sim: stim.TableauSimulator, basis: str, qubit: int) -> int:
    """Measure ``qubit`` in the named basis, returning the eigenvalue."""
    if basis == "X":
        sim.h(qubit)
    elif basis == "Y":
        sim.s_dag(qubit)
        sim.h(qubit)
    return -1 if sim.measure(qubit) else +1


@dataclass
class InterceptResend:
    """Measure each probe in a random basis and resend what was found."""

    name: str = "intercept_resend"
    reference: str = "derived"
    threat_class: str = ThreatClass.CHANNEL_MANIPULATION

    def run(self, spec: SchemeSpec, trials: int, seed: int) -> AttackResult:
        """Attack ``trials`` probes and count how often the tampering shows."""
        rng = np.random.default_rng(seed)
        detected = 0

        for trial in range(trials):
            true_basis = BASES[int(rng.integers(0, 3))]
            true_eigenvalue = 1 if rng.integers(0, 2) == 0 else -1
            attacker_basis = BASES[int(rng.integers(0, 3))]

            sim = stim.TableauSimulator(seed=seed * 100003 + trial)
            _prepare(sim, true_basis, true_eigenvalue, 0)

            # Attacker measures in its chosen basis and resends that eigenstate.
            found = _measure(sim, attacker_basis, 0)
            sim.reset(0)
            _prepare(sim, attacker_basis, found, 0)

            # Legitimate party measures in the basis the probe was prepared in.
            observed = _measure(sim, true_basis, 0)
            if observed != true_eigenvalue:
                detected += 1

        return AttackResult(
            attack=self.name,
            trials=trials,
            successes=trials - detected,
            verification_accepted=False,
            mismatch_rate=detected / trials if trials else 0.0,
            message_was_modified=False,
            attacker_used_key=False,
            detail={
                "detection_rate": detected / trials if trials else 0.0,
                "detection_theory": DETECTION_PER_PROBE,
                "escape_over_32_probes": escape_probability(32),
                "note": (
                    "Six-state figure 1/3, not BB84's 1/4: three mutually unbiased "
                    "bases mean the attacker guesses right only a third of the time."
                ),
            },
        )
