"""Layer 7 -- the decision table. Deterministic. No model anywhere.

**Two axes, and keeping them apart is the whole point.**

*Static axis* -- D1 alone. A pre-deployment protocol audit that fires on a
vulnerable specification whether or not anyone attacked it.

*Runtime axis* -- D2 through D5. Detectors watching an actual run.

Collapsing them produces an immediate contradiction: D1 fires on
``baseline_bell_aqs`` under perfectly honest traffic, so a single-axis table would
have to report an attack on a run where none occurred. Separating them lets honest
traffic on a vulnerable scheme say the true thing -- *the scheme is unsound, and
this particular run was clean* -- which is exactly what a certification tool should
say (ledger Q-5).

Unmapped runtime combinations return ``UNCLASSIFIED`` with the firing set
attached. An unmapped pattern is information: it means something happened that was
not anticipated, and swallowing it would be the worst behaviour available to an
assurance tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AttackClass(StrEnum):
    """What a firing pattern means."""

    NONE = "none"
    SCHEME_VULNERABLE_TO_RECEIVER_FORGERY = "scheme_vulnerable_to_receiver_forgery"
    CHANNEL_OR_OUTSIDER = "channel_or_outsider"
    REPLAY = "replay"
    IMPERSONATION = "impersonation"
    REPUDIATION = "repudiation"
    MALICIOUS_ARBITRATOR = "malicious_arbitrator"
    UNCLASSIFIED = "unclassified"


#: Runtime firing pattern -> attack class. D1 never appears here: it is the other
#: axis, and mixing it in is the error this table exists to avoid.
RUNTIME_DECISION_TABLE: dict[frozenset[str], AttackClass] = {
    frozenset(): AttackClass.NONE,
    frozenset({"D2"}): AttackClass.CHANNEL_OR_OUTSIDER,
    frozenset({"D3"}): AttackClass.REPUDIATION,
    frozenset({"D4"}): AttackClass.REPLAY,
    frozenset({"D5"}): AttackClass.IMPERSONATION,
    frozenset({"D3", "D5"}): AttackClass.MALICIOUS_ARBITRATOR,
}

#: Static firing pattern -> what the audit concluded about the scheme itself.
STATIC_DECISION_TABLE: dict[frozenset[str], AttackClass] = {
    frozenset(): AttackClass.NONE,
    frozenset({"D1"}): AttackClass.SCHEME_VULNERABLE_TO_RECEIVER_FORGERY,
}


@dataclass(frozen=True)
class Classification:
    """The two-axis verdict.

    Attributes:
        static_class: What the pre-deployment audit concluded about the scheme.
        runtime_class: What the runtime detectors concluded about this run.
        static_firing: Which static detectors fired.
        runtime_firing: Which runtime detectors fired.
        explanation: Why, in words. Never empty.
    """

    static_class: AttackClass
    runtime_class: AttackClass
    static_firing: frozenset[str] = field(default_factory=frozenset)
    runtime_firing: frozenset[str] = field(default_factory=frozenset)
    explanation: str = ""

    @property
    def is_clean(self) -> bool:
        """Both axes clear. The only state that means "nothing to report"."""
        return self.static_class is AttackClass.NONE and self.runtime_class is AttackClass.NONE


def classify(static_firing: set[str], runtime_firing: set[str]) -> Classification:
    """Map firing patterns to attack classes. Total: no input raises.

    Args:
        static_firing: Static detector ids that fired (only ``D1`` is static).
        runtime_firing: Runtime detector ids that fired (``D2``..``D5``).
    """
    static_key = frozenset(static_firing)
    runtime_key = frozenset(runtime_firing)

    static_class = STATIC_DECISION_TABLE.get(static_key, AttackClass.UNCLASSIFIED)
    runtime_class = RUNTIME_DECISION_TABLE.get(runtime_key, AttackClass.UNCLASSIFIED)

    parts = []
    if static_class is AttackClass.NONE:
        parts.append("Static audit: no structural vulnerability found in the specification.")
    elif static_class is AttackClass.UNCLASSIFIED:
        parts.append(
            f"Static audit produced an unmapped firing pattern {sorted(static_key)}. "
            "Surfacing it rather than defaulting, because an unanticipated pattern is "
            "information."
        )
    else:
        parts.append(
            "Static audit: the scheme is vulnerable by construction, independently of "
            "any run."
        )

    if runtime_class is AttackClass.NONE:
        parts.append("Runtime: no detector fired on this run.")
    elif runtime_class is AttackClass.UNCLASSIFIED:
        parts.append(
            f"Runtime produced an unmapped firing pattern {sorted(runtime_key)}, which "
            "no anticipated attack class explains. Investigate rather than dismiss."
        )
    else:
        parts.append(
            f"Runtime: detectors {sorted(runtime_key)} fired, classified as "
            f"{runtime_class.value}."
        )

    if static_class is not AttackClass.NONE and runtime_class is AttackClass.NONE:
        parts.append(
            "Note the combination: a structurally unsound scheme with a clean run. The "
            "Choi-class forgery passes verification, so a runtime detector watching "
            "accept/reject outcomes has nothing to see. That is the point of running "
            "the static audit at all."
        )

    return Classification(
        static_class=static_class,
        runtime_class=runtime_class,
        static_firing=static_key,
        runtime_firing=runtime_key,
        explanation=" ".join(parts),
    )
