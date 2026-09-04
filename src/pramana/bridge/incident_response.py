"""Detector -> severity -> recommended action.

A detection framework with no response path is incomplete, and this closes that
gap. Each entry names what a lab or CA should actually do, not a severity colour.

The severities are deliberately conservative on the static axis: D1 firing means
the *scheme* is unsound, which is a certification decision rather than an incident.
Treating it as a runtime alert would misdirect the response.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IncidentAction:
    """What to do when a detector fires."""

    detector: str
    severity: str
    action: str
    rationale: str


INCIDENT_RESPONSE: dict[str, IncidentAction] = {
    "D1": IncidentAction(
        detector="D1",
        severity="critical",
        action="Do not certify. Halt deployment and apply the suggested repair.",
        rationale=(
            "A static audit finding means the scheme is forgeable by construction, for "
            "every key and every run. No amount of runtime monitoring compensates, "
            "because the forgery passes verification. This is a certification "
            "decision, not an incident to triage."
        ),
    ),
    "D2": IncidentAction(
        detector="D2",
        severity="high",
        action="Halt signing on this channel, recalibrate on a known-clean link, alert the CA.",
        rationale=(
            "A probe error rate above baseline indicates channel manipulation, an "
            "outsider forgery attempt, or an unauthorised verifier. The SPRT reports a "
            "bounded false-alarm rate, so the alert is actionable rather than advisory."
        ),
    ),
    "D3": IncidentAction(
        detector="D3",
        severity="high",
        action="Freeze the disputed signature, escalate to the arbitrator, log both verdicts.",
        rationale=(
            "Verifiers disagreeing on one signature is repudiation. The dispute needs "
            "an arbitrator, not a threshold, and both verdicts must be preserved as "
            "evidence."
        ),
    ),
    "D4": IncidentAction(
        detector="D4",
        severity="critical",
        action="Reject the submission and revoke the reused credential.",
        rationale=(
            "An entangled-pair index cannot legitimately recur. The check is exact, so "
            "there is no false-positive path to consider before acting."
        ),
    ),
    "D5": IncidentAction(
        detector="D5",
        severity="critical",
        action="Revoke the claimed signer's certificate and alert the CA.",
        rationale=(
            "Identity-bound probes failing means the signing party does not hold the "
            "claimed key material. Escape probability falls as 2**-d, so a sustained "
            "failure is not chance."
        ),
    ),
}


def action_for(detector: str) -> IncidentAction:
    """The response entry for a detector.

    Raises:
        KeyError: No entry exists. Every detector must have one, and
            ``test_bridge.py`` asserts that.
    """
    if detector not in INCIDENT_RESPONSE:
        raise KeyError(
            f"no incident-response entry for {detector!r}. Every detector needs one: a "
            "detection with no response path is an incomplete framework."
        )
    return INCIDENT_RESPONSE[detector]
