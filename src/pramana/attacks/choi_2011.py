"""Choi existential forgery, variant 1. The project's defining demonstration.

**Source:** Choi, Chang, Hong, *Security problem on arbitrated quantum signature
schemes*, Phys. Rev. A **84**, 062330 (2011), section III A. Attack steps taken
from ``docs/derivations.md`` section 5.6, written from the paper by a team member,
not reconstructed from a general description.

**The attack.** Choi Eq. (2): the receiver applies a Pauli ``Q`` to both the
message register and the quantum part of the signature. Because every operator in
the signing path is a Pauli, ``Q`` commutes with the whole path up to a global
phase, and

    (Q|P>, Q|S(P)>)  ==  (|P'>, |S(P')>)   up to global phase

so the pair is a *valid* signature on the modified message. Choi: **"This type of
attack is always successful."** Success probability 1, no key required.

**Why it runs through the real verifier.** The attack calls
``sign_and_verify_round`` with its ``forge`` hook rather than reimplementing the
protocol. A forgery demonstrated against a reimplementation of verification would
prove nothing about the verification this project actually ships.

**Which ``Q``.** D1's witness search predicts exactly which Paulis commute up to
phase with every encryption and rotation operator, so the attack takes its forging
operator from that prediction. Only ``X`` and ``Y`` modify a classical message
(``Z`` and ``I`` leave a computational-basis state unchanged up to phase), so a
witness that forges must be one of those.

**Classical-message caveat -- ledger V-42.** For a classical message, verification
reads only the ``Z`` observable. The attack therefore succeeds whenever the net
operator deterministically flips the computational basis, which is a **strictly
weaker** condition than commuting up to global phase. ``pauli_witness_free_aqs``
has *zero* Pauli witnesses and is still forged by ``Q = Y``. D1's predicate
decides the general-quantum-message class that Choi analyses; it does not decide
the classical-message case, and this attack reports which situation it is in via
``predicted_by_d1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.attacks.base import AttackResult, ThreatClass
from pramana.protocol.signing import sign_and_verify_round
from pramana.spec.schema import SchemeSpec

#: Paulis that actually modify a classical message. Z and I do not.
MESSAGE_MODIFYING = ("X", "Y")


def select_forging_operator(spec: SchemeSpec) -> tuple[str, bool]:
    """Choose ``Q``, and report whether D1 predicted it.

    Returns:
        ``(operator, predicted_by_d1)``. When D1 finds a message-modifying
        witness, that witness is used. When it finds none, ``Y`` is tried anyway
        and the caller is told D1 did not predict it -- see the classical-message
        caveat in the module docstring.
    """
    witnesses = spec.forging_witnesses()
    for candidate in MESSAGE_MODIFYING:
        if candidate in witnesses:
            return candidate, True
    return "Y", False


@dataclass
class Choi2011:
    """Choi section III A variant 1 existential forgery."""

    forge: str | None = None
    name: str = "choi_2011"
    reference: str = "Choi, Chang, Hong, Phys. Rev. A 84, 062330 (2011), section III A"
    threat_class: str = ThreatClass.RECEIVER_FORGERY

    def run(self, spec: SchemeSpec, trials: int, seed: int) -> AttackResult:
        """Forge ``trials`` message-signature pairs.

        Success is counted only when verification accepted **and** the message
        actually changed. An attack that leaves the message identical has forged
        nothing, however cleanly it verifies.
        """
        operator, predicted = (
            (self.forge, self.forge in spec.forging_witnesses())
            if self.forge is not None
            else select_forging_operator(spec)
        )
        length = spec.signature.length_qubits
        message = tuple((i * 7 + 3) % 2 for i in range(length))

        accepted = 0
        total_mismatches = 0
        for trial in range(trials):
            _signed, ok, mismatches = sign_and_verify_round(
                message, spec, seed=seed + trial, forge=operator
            )
            accepted += int(ok)
            total_mismatches += len(mismatches)

        modified = operator in MESSAGE_MODIFYING
        return AttackResult(
            attack=self.name,
            trials=trials,
            successes=accepted if modified else 0,
            verification_accepted=accepted == trials and trials > 0,
            mismatch_rate=total_mismatches / (trials * length) if trials else 0.0,
            message_was_modified=modified,
            attacker_used_key=False,
            detail={
                "forging_operator": operator,
                "predicted_by_d1": predicted,
                "d1_witnesses": list(spec.forging_witnesses()),
                "reference": "Choi section III A, Eq. (2)",
                "note": (
                    "D1 predicted this operator."
                    if predicted
                    else "D1 found no witness; this succeeds only because a classical "
                    "message is verified in the Z basis alone (ledger V-42)."
                ),
            },
        )
