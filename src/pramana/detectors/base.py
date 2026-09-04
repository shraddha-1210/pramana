"""The detector interface and the verdict every detector must produce.

**The explanation field is not optional.** This is a certification tool: a verdict
a lab cannot justify is not an output, it is a liability. Every verdict carries the
specific evidence that triggered it.

**Two axes, kept separate.** D1 is a *static pre-deployment protocol audit*: it
runs on the specification alone, before any simulation, and it fires on a
vulnerable scheme whether or not anyone attacked. D2-D5 are *runtime* detectors
watching an actual run. Reporting D1 as runtime detection invites a correct
objection, and the decision table keeps the axes apart for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pramana.engines.base import RoundResults
from pramana.spec.schema import SchemeSpec


class DetectorKind(StrEnum):
    """Which axis a detector belongs to."""

    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class Verdict:
    """One detector's finding.

    Attributes:
        detector: "D1".."D5".
        kind: Static audit or runtime detection.
        fired: Whether the detector is reporting a problem.
        applicable: Whether the detector could run at all. A detector that does
            not apply must say so rather than silently reporting "clear" --
            "not applicable" and "no problem found" are different claims.
        confidence: Exactness for algebraic and ledger checks; a probability-like
            score for statistical ones.
        evidence: The specific data that triggered the verdict.
        explanation: Human-readable justification. Never empty.
        suggested_fix: Repair advice, when the detector has one.
    """

    detector: str
    kind: DetectorKind
    fired: bool
    applicable: bool
    confidence: float
    evidence: dict[str, object] = field(default_factory=dict)
    explanation: str = ""
    suggested_fix: str = ""

    def __post_init__(self) -> None:
        if not self.explanation.strip():
            raise ValueError(
                f"{self.detector} produced a verdict with no explanation. Every verdict "
                "in a certification tool must be justifiable to a lab."
            )


class Detector(Protocol):
    """A detector."""

    id: str
    kind: DetectorKind

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None) -> Verdict:
        """Assess a scheme, and a run when there is one."""
        ...


def not_applicable(detector: str, kind: DetectorKind, reason: str) -> Verdict:
    """A verdict that says the detector could not run. Never a silent pass."""
    return Verdict(
        detector=detector,
        kind=kind,
        fired=False,
        applicable=False,
        confidence=0.0,
        explanation=reason,
    )
