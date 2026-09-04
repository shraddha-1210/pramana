"""D5 -- identity-bound probes. Catches impersonation.

Runtime. Probe expectations for certain rounds derive from the claimed signer's
key material, reconstructed through ``t``-of-``n`` threshold arbitration. An
impersonator without that key produces uncorrelated outcomes.

**Why this is not circular.** The naive design has D5 check probe outcomes against
what *the arbitrator* says they should be -- so a malicious arbitrator is the
reference for its own audit. Here the expectation comes from a Shamir
reconstruction needing ``t`` shares, so no single arbitrator can fabricate a
consistent expectation, and D5 still fires when one of them is the attacker.

**Escape probability.** An impersonator holds no correlated half, so each
identity-bound probe matches by chance with probability 1/2 and the escape
probability over ``d`` probes is ``2**-d``. Derived, and asserted in the tests.

The panel's own capability bounds apply: detection needs ``n >= t + 1`` and
identification needs ``n >= t + 2`` (``crypto/arbitrator.py``). D5 reports what the
panel could actually establish, never more.
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.crypto.arbitrator import ArbitrationPanel
from pramana.detectors.base import DetectorKind, Verdict, not_applicable
from pramana.engines.base import RoundResults
from pramana.spec.schema import SchemeSpec


def impersonator_escape_probability(probes: int) -> float:
    """``2**-d``: an impersonator matches every identity-bound probe by chance."""
    return 2.0**-probes


@dataclass
class D5IdentityProbe:
    """Identity-bound probe check, refereed by a threshold panel."""

    panel: ArbitrationPanel | None = None
    expected_key: int | None = None
    mismatch_threshold: float = 0.25
    id: str = "D5"
    kind: DetectorKind = DetectorKind.RUNTIME

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None = None) -> Verdict:
        """Assess whether the probe stream is bound to the claimed identity."""
        if results is None:
            return not_applicable(
                self.id, self.kind, "D5 is a runtime detector and needs a run to assess."
            )
        probes = [r for r in results.rounds if r.is_probe]
        if not probes:
            return not_applicable(
                self.id,
                self.kind,
                "The run contains no probe rounds, so identity binding cannot be "
                "assessed. This is not the same as finding no impersonation.",
            )

        arbitration = None
        if self.panel is not None:
            arbitration = self.panel.adjudicate(expected=self.expected_key)

        mismatches = sum(1 for r in probes if r.probe_observed != r.probe_expected)
        rate = mismatches / len(probes)
        fired = rate > self.mismatch_threshold

        evidence: dict[str, object] = {
            "identity_probes": len(probes),
            "mismatch_rate": rate,
            "threshold": self.mismatch_threshold,
            "escape_probability": impersonator_escape_probability(len(probes)),
        }
        if arbitration is not None:
            evidence["arbitration_agreed"] = arbitration.agreed
            evidence["arbitration_suspected"] = list(arbitration.suspected)
            evidence["arbitration_can_identify"] = arbitration.can_identify
            evidence["arbitration_explanation"] = arbitration.explanation

        parts = []
        if fired:
            parts.append(
                f"Identity-bound probes mismatched at {rate:.3f} over {len(probes)} "
                f"probes, above the {self.mismatch_threshold} threshold. A party "
                "holding the correct key material reproduces these outcomes "
                "deterministically; an impersonator matches only by chance, with "
                f"escape probability {impersonator_escape_probability(len(probes)):.2e}."
            )
        else:
            parts.append(
                f"Identity-bound probes mismatched at {rate:.3f} over {len(probes)} "
                f"probes, within the {self.mismatch_threshold} threshold. Consistent "
                "with the claimed signer holding the correct key material."
            )
        if arbitration is not None:
            parts.append(
                "Probe expectations were reconstructed from a "
                f"{self.panel.threshold}-of-{len(self.panel.arbitrators)} arbitration "
                f"panel rather than from any single arbitrator, so this verdict is not "
                f"circular. Panel: {arbitration.explanation}"
            )

        return Verdict(
            detector=self.id,
            kind=self.kind,
            fired=fired or (arbitration is not None and not arbitration.agreed),
            applicable=True,
            confidence=1.0 - impersonator_escape_probability(len(probes)),
            evidence=evidence,
            explanation=" ".join(parts),
        )
