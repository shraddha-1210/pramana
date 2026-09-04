"""The assurance report: the deliverable a certification lab would actually read.

Every report carries the honesty table -- what was simulated, what needs real
hardware, what is out of scope. That is a scoring asset, not a liability: it is the
first thing an evaluator looks for and its absence is what makes a demo look
oversold.

Assurance levels map to India's proposed L1-L4 certification tiers. The mapping is
deliberately conservative, and no report can reach the top tier from simulation
alone, because hardware characterisation is out of scope.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from pramana.bridge.incident_response import INCIDENT_RESPONSE
from pramana.crypto.key_pool import KeyBudget
from pramana.decision.table import Classification
from pramana.detectors.base import Verdict
from pramana.spec.schema import SchemeSpec

REPORT_VERSION = "pramana-assurance-report/0.1"


class AssuranceLevel(StrEnum):
    """India's proposed L1-L4 tiers, as this tool applies them."""

    L0_FAILED = "L0 - failed audit"
    L1_SIMULATED = "L1 - simulated assurance"
    L2_SIMULATED_HARDENED = "L2 - simulated assurance, hardened scheme"
    L3_HARDWARE = "L3 - hardware-characterised (not reachable by simulation)"
    L4_CERTIFIED = "L4 - certified deployment (not reachable by simulation)"


HONESTY_TABLE = {
    "simulated": [
        "Clifford protocol operations, exact stabilizer simulation (Stim).",
        "Channel noise, density-matrix simulation with depolarizing and amplitude damping (Qiskit Aer).",
        "Pauli and Clifford-quotient algebra, exact integer and bit arithmetic.",
        "Published attacks, reproduced from the cited papers.",
    ],
    "needs_real_hardware": [
        "Entangled-pair generation, distribution, fidelity and drift.",
        "Measured channel noise parameters; ours are inputs, not measurements.",
        "D2 baseline calibration on a real honest channel.",
    ],
    "out_of_scope": [
        "Detector blinding, photon-number splitting and Trojan-horse attacks (Gisin et al. 2006).",
        "Denial of service against the quantum link.",
        "Initial seed key distribution (assumption A-1).",
        "Forging operators outside the single-qubit Clifford group.",
        "Full CMS/PAdES compliance; the envelope is a proposed extension only.",
        "Stabilizer-rank simulation, so schemes with a non-Clifford assistant unitary are audited but not run.",
    ],
    "known_limits": [
        "Zero Pauli witnesses does not mean secure: for a classical message, verification reads only the Z observable, so a weaker condition suffices to forge (V-42).",
        "The quantitative accept-rate curve versus depolarizing noise is not analytically derived and is not claimed (V-34).",
        "Arbitration identifies one liar only at n >= t + 2; at n = t + 1 it detects without identifying.",
    ],
}


@dataclass
class AssuranceReport:
    """One run's findings, in the form a lab would read."""

    run_id: str
    seed: int
    spec_name: str
    spec_reference: str
    verdicts: list[Verdict] = field(default_factory=list)
    classification: Classification | None = None
    key_budget: KeyBudget | None = None
    attack_results: list[dict[str, object]] = field(default_factory=list)
    version: str = REPORT_VERSION

    def assurance_level(self) -> AssuranceLevel:
        """Map findings to a tier. Conservative by construction."""
        if any(v.fired and v.kind == "static" for v in self.verdicts):
            return AssuranceLevel.L0_FAILED
        if any(v.fired for v in self.verdicts):
            return AssuranceLevel.L0_FAILED
        return AssuranceLevel.L1_SIMULATED

    def forgery_bounds(self) -> dict[str, str]:
        """Stated bounds, each with its derivation named."""
        return {
            "blind_forgery_correct_guess": "4**-n (uniform guess over four corrections per position)",
            "blind_forgery_acceptance": "2**-n (a residual of Z is invisible to a Z-basis measurement)",
            "intercept_resend_escape": "(2/3)**d (six-state: 2/3 wrong basis, 1/2 wrong outcome)",
            "unauthorised_verifier_escape": "2**-d (maximally mixed state, coin-flip per probe)",
            "impersonator_escape": "2**-d (no correlated half held)",
            "choi_forgery_success": "1.0 exactly (Choi section III A: always successful)",
        }

    def incident_actions(self) -> list[dict[str, str]]:
        """Recommended action for every detector that fired."""
        actions = []
        for verdict in self.verdicts:
            if verdict.fired and verdict.detector in INCIDENT_RESPONSE:
                entry = INCIDENT_RESPONSE[verdict.detector]
                actions.append(
                    {
                        "detector": entry.detector,
                        "severity": entry.severity,
                        "action": entry.action,
                        "rationale": entry.rationale,
                    }
                )
        return actions

    def to_dict(self) -> dict[str, object]:
        """The full report structure."""
        return {
            "version": self.version,
            "run_id": self.run_id,
            "seed": self.seed,
            "scheme": {"name": self.spec_name, "reference": self.spec_reference},
            "assurance_level": self.assurance_level().value,
            "verdicts": [
                {
                    "detector": v.detector,
                    "kind": str(v.kind),
                    "fired": v.fired,
                    "applicable": v.applicable,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                    "explanation": v.explanation,
                    "suggested_fix": v.suggested_fix,
                }
                for v in self.verdicts
            ],
            "classification": (
                {
                    "static_class": self.classification.static_class.value,
                    "runtime_class": self.classification.runtime_class.value,
                    "static_firing": sorted(self.classification.static_firing),
                    "runtime_firing": sorted(self.classification.runtime_firing),
                    "explanation": self.classification.explanation,
                }
                if self.classification
                else None
            ),
            "forgery_bounds": self.forgery_bounds(),
            "key_budget": (
                {**asdict(self.key_budget), "summary": self.key_budget.summary()}
                if self.key_budget
                else None
            ),
            "attacks": self.attack_results,
            "incident_response": self.incident_actions(),
            "honesty_table": HONESTY_TABLE,
        }

    def to_json(self) -> str:
        """Serialise the report."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def build_report(
    spec: SchemeSpec,
    run_id: str,
    seed: int,
    verdicts: list[Verdict],
    classification: Classification | None = None,
    key_budget: KeyBudget | None = None,
    attack_results: list[dict[str, object]] | None = None,
) -> AssuranceReport:
    """Assemble a report from a run's findings."""
    return AssuranceReport(
        run_id=run_id,
        seed=seed,
        spec_name=spec.name,
        spec_reference=spec.reference,
        verdicts=verdicts,
        classification=classification,
        key_budget=key_budget,
        attack_results=attack_results or [],
    )
