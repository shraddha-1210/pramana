"""D2 -- sequential probability ratio test on probe rounds.

Runtime. The workhorse: catches outsider forgery, unauthorised verification and
channel manipulation, all of which perturb the probe stream.

**Wald's SPRT.** Two hypotheses about the probe error rate:

* ``H0``: the rate is the calibrated baseline ``p0``.
* ``H1``: the rate is the attack threshold ``p1 > p0``.

Accumulate the log-likelihood ratio per probe and stop when it crosses a bound:

    accept H1 (attack)  when  LLR >= ln((1 - beta) / alpha)
    accept H0 (honest)  when  LLR <= ln(beta / (1 - alpha))

with ``alpha`` the target false-alarm rate and ``beta`` the target miss rate. The
per-probe increment is ``ln(p1/p0)`` on a mismatch and ``ln((1-p1)/(1-p0))`` on a
match. Derived in ``docs/derivations.md`` section 8.

**Why SPRT rather than a fixed threshold.** Error rates you can *state*, and
evidence that accumulates -- so an attacker sitting just under a fixed threshold is
still caught, only later.

**The baseline comes from the noise engine. Never the Clifford engine.**
This is not a preference. The Clifford engine is noiseless, so it yields
``p0 = 0``, and ``ln(p1/p0)`` diverges: the first probe error would decide the
test with infinite confidence and the stated ``alpha`` would be meaningless. The
noise engine models an actual channel and produces a positive rate. The scheme's
declared ``probes.p_min`` is a floor beneath which no measured baseline is
accepted, because a floor hidden inside the detector would invalidate the
false-alarm rate it advertises.

**Calibration poisoning.** An attacker present during calibration is absorbed into
the baseline and becomes invisible. Three mitigations, all implemented:

1. Assumption A-3, stated in the threat model.
2. Periodic recalibration, so one poisoned session does not persist.
3. A cross-check against theory: the channel's physical parameters give an
   expected error rate, and a measured baseline exceeding it by more than a
   configured margin is flagged. This is the strong defence, and it is available
   precisely because the model is analytic rather than learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pramana.detectors.base import DetectorKind, Verdict, not_applicable
from pramana.engines.base import RoundResults
from pramana.engines.noise import NoiseEngine
from pramana.spec.schema import SchemeSpec


@dataclass(frozen=True)
class SprtBounds:
    """Wald's decision boundaries."""

    upper: float
    lower: float

    @classmethod
    def from_error_rates(cls, alpha: float, beta: float) -> SprtBounds:
        """``ln((1-beta)/alpha)`` and ``ln(beta/(1-alpha))``."""
        if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError(
                f"alpha and beta must lie strictly in (0, 1); got {alpha}, {beta}."
            )
        return cls(
            upper=math.log((1.0 - beta) / alpha),
            lower=math.log(beta / (1.0 - alpha)),
        )


@dataclass(frozen=True)
class SprtTrace:
    """The running log-likelihood ratio, for the timeline view."""

    values: tuple[float, ...]
    decision: str
    decided_at: int | None


def calibrate_baseline(
    spec: SchemeSpec,
    *,
    depolarizing_p: float,
    damping_gamma: float,
    rounds: int,
    seed: int,
) -> float:
    """Measure ``p0`` on a known-honest channel using the **noise** engine.

    Returns the measured probe error rate, floored at the scheme's declared
    ``probes.p_min``. The floor is part of the specification, not a hidden
    constant, so the advertised false-alarm rate stays meaningful.
    """
    engine = NoiseEngine(depolarizing_p=depolarizing_p, damping_gamma=damping_gamma)
    results = engine.run_rounds(spec, rounds=rounds, seed=seed)
    measured = 1.0 - results.accept_rate
    return max(measured, spec.probes.p_min)


def theoretical_error_bound(depolarizing_p: float, damping_gamma: float) -> float:
    """An analytic upper bound on the honest-channel error rate.

    Each noisy channel can only push a measured observable towards uniform, so a
    crude but sound bound on the per-round error contributed by the two channels
    is their sum, capped at 1/2 -- the fully randomised outcome. It is deliberately
    loose: its job is to catch a baseline that is *implausibly high*, which is what
    calibration poisoning looks like, not to predict the rate precisely.

    An exact closed form for this circuit has not been derived (ledger V-34), so
    none is claimed here.
    """
    return min(0.5, depolarizing_p + damping_gamma)


def run_sprt(
    mismatches: list[bool], p0: float, p1: float, bounds: SprtBounds
) -> SprtTrace:
    """Accumulate the LLR over a probe stream and report where it stopped."""
    if not 0.0 < p0 < p1 < 1.0:
        raise ValueError(
            f"need 0 < p0 < p1 < 1 for a well-posed test; got p0={p0}, p1={p1}. "
            "A zero baseline makes the likelihood ratio degenerate -- calibrate "
            "against the noise engine, not the Clifford engine."
        )
    on_mismatch = math.log(p1 / p0)
    on_match = math.log((1.0 - p1) / (1.0 - p0))

    llr = 0.0
    values: list[float] = []
    decision, decided_at = "continue", None
    for index, mismatch in enumerate(mismatches):
        llr += on_mismatch if mismatch else on_match
        values.append(llr)
        if decision == "continue":
            if llr >= bounds.upper:
                decision, decided_at = "attack", index
            elif llr <= bounds.lower:
                decision, decided_at = "honest", index
    return SprtTrace(values=tuple(values), decision=decision, decided_at=decided_at)


@dataclass
class D2ProbeSprt:
    """Sequential test on the probe stream."""

    p0: float = 0.02
    p1: float = 0.10
    alpha: float = 0.01
    beta: float = 0.01
    theory_margin: float = 2.0
    id: str = "D2"
    kind: DetectorKind = DetectorKind.RUNTIME

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None = None) -> Verdict:
        """Run the SPRT over the probe rounds of ``results``."""
        if results is None:
            return not_applicable(
                self.id, self.kind, "D2 is a runtime detector and needs a run to assess."
            )
        probes = [r for r in results.rounds if r.is_probe]
        if not probes:
            return not_applicable(
                self.id,
                self.kind,
                "The run contains no probe rounds, so there is no probe stream to test. "
                "This is not the same as finding no attack.",
            )

        floor = max(self.p0, spec.probes.p_min)
        bounds = SprtBounds.from_error_rates(self.alpha, self.beta)
        mismatches = [r.probe_observed != r.probe_expected for r in probes]
        trace = run_sprt(mismatches, floor, self.p1, bounds)

        observed_rate = sum(mismatches) / len(mismatches)
        fired = trace.decision == "attack"
        evidence = {
            "probes": len(probes),
            "observed_error_rate": observed_rate,
            "p0": floor,
            "p1": self.p1,
            "alpha": self.alpha,
            "beta": self.beta,
            "upper_bound": bounds.upper,
            "lower_bound": bounds.lower,
            "decision": trace.decision,
            "decided_at_probe": trace.decided_at,
            "llr_final": trace.values[-1] if trace.values else 0.0,
        }

        if fired:
            explanation = (
                f"The probe error rate is {observed_rate:.3f} over {len(probes)} probes. "
                f"The log-likelihood ratio crossed the upper Wald bound "
                f"({bounds.upper:.2f}) at probe {trace.decided_at}, so H1 (error rate "
                f"{self.p1}) is accepted over H0 (baseline {floor}) at a target "
                f"false-alarm rate of {self.alpha}."
            )
        elif trace.decision == "honest":
            explanation = (
                f"The probe error rate is {observed_rate:.3f} over {len(probes)} probes. "
                f"The ratio crossed the lower bound ({bounds.lower:.2f}) at probe "
                f"{trace.decided_at}: consistent with the calibrated baseline "
                f"{floor} at a target miss rate of {self.beta}."
            )
        else:
            explanation = (
                f"The probe error rate is {observed_rate:.3f} over {len(probes)} probes, "
                f"and the ratio ({trace.values[-1] if trace.values else 0.0:.2f}) has "
                "not reached either bound. The test is undecided; more probes are "
                "needed. This is not a clean bill of health."
            )

        return Verdict(
            detector=self.id,
            kind=self.kind,
            fired=fired,
            applicable=True,
            confidence=1.0 - self.alpha if fired else 1.0 - self.beta,
            evidence=evidence,
            explanation=explanation,
        )

    def check_calibration_against_theory(
        self, measured_p0: float, depolarizing_p: float, damping_gamma: float
    ) -> Verdict:
        """Flag a baseline that exceeds what the channel physics allows.

        The defence against calibration poisoning that a learned baseline could
        not offer: there is an analytic model of what "normal" should be, so a
        baseline inflated by an attacker present during calibration is visible as
        a physical implausibility rather than as a statistical anomaly.
        """
        bound = theoretical_error_bound(depolarizing_p, damping_gamma)
        limit = bound * self.theory_margin
        poisoned = measured_p0 > limit
        return Verdict(
            detector=f"{self.id}-calibration",
            kind=self.kind,
            fired=poisoned,
            applicable=True,
            confidence=1.0,
            evidence={
                "measured_p0": measured_p0,
                "theoretical_bound": bound,
                "margin": self.theory_margin,
                "limit": limit,
            },
            explanation=(
                f"Measured baseline {measured_p0:.4f} exceeds {self.theory_margin}x the "
                f"theoretical bound {bound:.4f} for a channel with depolarizing "
                f"{depolarizing_p} and damping {damping_gamma}. A baseline the physics "
                "cannot produce suggests an attacker was present during calibration "
                "(assumption A-3). Recalibrate on a known-clean channel."
                if poisoned
                else (
                    f"Measured baseline {measured_p0:.4f} is within {self.theory_margin}x "
                    f"the theoretical bound {bound:.4f}, so the calibration is physically "
                    "plausible."
                )
            ),
        )
