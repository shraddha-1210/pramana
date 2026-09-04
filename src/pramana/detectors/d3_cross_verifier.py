"""D3 -- cross-verifier consistency. Catches repudiation.

Runtime. Repudiation means the signer makes two verifiers disagree about the same
signature. By definition a single verifier watching itself cannot detect that, so
this is a consistency check across verifiers, not a threshold on one.

**Scope, stated honestly.** This applies to multi-party settings -- banking
consortia, multi-CA arrangements -- and not to the common one-signer-one-verifier
case. With a single verifier D3 reports *inapplicable*, never a silent pass:
"cannot assess" and "found nothing" are different claims, and a certification tool
must not blur them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pramana.detectors.base import DetectorKind, Verdict, not_applicable
from pramana.engines.base import RoundResults
from pramana.spec.schema import SchemeSpec


@dataclass
class D3CrossVerifier:
    """Compare accept/reject across independent verifiers."""

    verifier_results: dict[str, RoundResults] = field(default_factory=dict)
    id: str = "D3"
    kind: DetectorKind = DetectorKind.RUNTIME

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None = None) -> Verdict:
        """Assess agreement between the verifiers supplied at construction."""
        if len(self.verifier_results) < 2:
            return not_applicable(
                self.id,
                self.kind,
                f"D3 needs at least two independent verifiers to compare; "
                f"{len(self.verifier_results)} supplied. Repudiation is by definition "
                "invisible to a single verifier watching itself, so this is reported "
                "as inapplicable rather than as no problem found.",
            )

        names = sorted(self.verifier_results)
        per_round: dict[str, list[bool]] = {
            name: [r.accepted for r in self.verifier_results[name].rounds] for name in names
        }
        lengths = {len(v) for v in per_round.values()}
        if len(lengths) != 1:
            return not_applicable(
                self.id,
                self.kind,
                f"Verifiers reported different numbers of rounds ({sorted(lengths)}); "
                "they are not looking at the same run and cannot be compared.",
            )

        rounds = lengths.pop()
        disagreements = [
            index
            for index in range(rounds)
            if len({per_round[name][index] for name in names}) > 1
        ]
        fired = bool(disagreements)
        return Verdict(
            detector=self.id,
            kind=self.kind,
            fired=fired,
            applicable=True,
            confidence=1.0,  # exact comparison, not a statistical test
            evidence={
                "verifiers": names,
                "rounds_compared": rounds,
                "disagreement_count": len(disagreements),
                "first_disagreements": disagreements[:10],
            },
            explanation=(
                f"Verifiers {names} disagreed on {len(disagreements)} of {rounds} rounds "
                f"(first at {disagreements[:5]}). Two honest verifiers presented with the "
                "same signature must reach the same verdict, so disagreement means the "
                "signer produced a pair that verifies differently for different parties "
                "-- repudiation."
                if fired
                else (
                    f"Verifiers {names} agreed on all {rounds} rounds. No repudiation "
                    "signal in this run."
                )
            ),
        )
