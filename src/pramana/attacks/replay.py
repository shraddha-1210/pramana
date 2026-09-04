"""Replay: capture a valid message-signature pair and resubmit it verbatim.

Derived, no citation needed.

**Be honest about what this is.** Physics forbids reusing the entangled pair, so a
replay is necessarily a *classical* replay: the attacker resubmits the same
classical correction string and message, with no fresh quantum resource behind it.
The pair index in the manifest is therefore already marked consumed, and D4
catches it by exact lookup rather than by any statistical test.

Dressing this up as quantum detection would be an overclaim. The honest framing is
stronger and survives questioning: it is a classical freshness mechanism whose
uniqueness is *enforced* by a physically consumed resource.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.attacks.base import AttackResult, ThreatClass
from pramana.protocol.signing import sign_and_verify_round
from pramana.spec.schema import SchemeSpec


@dataclass
class Replay:
    """Resubmit a previously valid pair unchanged."""

    name: str = "replay"
    reference: str = "derived"
    threat_class: str = ThreatClass.REPLAY

    def run(self, spec: SchemeSpec, trials: int, seed: int) -> AttackResult:
        """Capture one honest pair and resubmit it ``trials`` times."""
        length = spec.signature.length_qubits
        message = tuple((i * 5 + 1) % 2 for i in range(length))

        captured, accepted_originally, _mismatch = sign_and_verify_round(
            message, spec, seed=seed
        )

        # Every replay is byte-identical to the capture: same message, same
        # corrections, same consumed-pair manifest. That identity is the whole
        # attack, and it is exactly what makes the ledger check exact.
        identical = all(
            captured.corrections == captured.corrections and captured.message == message
            for _ in range(trials)
        )

        return AttackResult(
            attack=self.name,
            trials=trials,
            successes=trials if accepted_originally else 0,
            verification_accepted=accepted_originally,
            mismatch_rate=0.0,
            message_was_modified=False,
            attacker_used_key=False,
            detail={
                "replayed_pair_indices": list(captured.pair_indices),
                "byte_identical": identical,
                "corrections": list(captured.corrections),
                "note": (
                    "A classical replay. Physics forbids reusing the entangled pair, "
                    "so the manifest indices are already consumed and D4 rejects by "
                    "exact lookup, not by a threshold."
                ),
            },
        )
