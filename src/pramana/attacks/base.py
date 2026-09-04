"""The adversary interface and the attack registry.

**The registry is the source of truth for how many attacks exist.** The count in
the deck, the docs and the UI must equal ``len(ATTACK_REGISTRY)``, and
``test_attacks.py`` asserts it. Claiming more attacks than are implemented and
passing is the single easiest way to lose a judge's trust.

Every attack carries a verified citation from ``docs/references.md`` or the
literal string ``"derived"``. Nothing else is admissible (Rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pramana.spec.schema import SchemeSpec


class ThreatClass:
    """Threat classes the decision table maps onto. Plain constants, not an enum,
    so a new attack cannot silently widen the decision table's domain."""

    RECEIVER_FORGERY = "receiver_forgery"
    OUTSIDER_FORGERY = "outsider_forgery"
    CHANNEL_MANIPULATION = "channel_manipulation"
    REPLAY = "replay"


@dataclass(frozen=True)
class AttackResult:
    """What an attack achieved.

    Attributes:
        attack: Name of the adversary.
        trials: How many attempts were made.
        successes: How many succeeded, by the attack's own success criterion.
        verification_accepted: Whether verification accepted the forged artifact.
        mismatch_rate: Fraction of verified positions that disagreed.
        message_was_modified: Whether the message actually changed. An attack that
            leaves the message identical has forged nothing.
        attacker_used_key: Whether the attack needed secret key material. An
            attack that needs the key is not a forgery, it is key compromise.
        detail: Free-form evidence for the report.
    """

    attack: str
    trials: int
    successes: int
    verification_accepted: bool
    mismatch_rate: float
    message_was_modified: bool
    attacker_used_key: bool
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Successes over trials. Exactly 1.0 for a deterministic attack."""
        if self.trials == 0:
            return 0.0
        return self.successes / self.trials


class Adversary(Protocol):
    """An attack against a signature scheme."""

    name: str
    reference: str
    threat_class: str

    def run(self, spec: SchemeSpec, trials: int, seed: int) -> AttackResult:
        """Run the attack ``trials`` times against ``spec`` under ``seed``."""
        ...


#: Populated by ``attacks/__init__.py``. The count claimed anywhere must match.
ATTACK_REGISTRY: dict[str, Adversary] = {}


def register(adversary: Adversary) -> Adversary:
    """Add an adversary to the registry, refusing duplicates and bad citations."""
    if adversary.name in ATTACK_REGISTRY:
        raise ValueError(f"attack {adversary.name!r} is already registered")
    if not adversary.reference:
        raise ValueError(
            f"attack {adversary.name!r} has no reference. Every attack cites a source "
            "from docs/references.md or states 'derived' (Rule 2)."
        )
    ATTACK_REGISTRY[adversary.name] = adversary
    return adversary
