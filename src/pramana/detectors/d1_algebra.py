"""D1 -- static algebraic audit of the scheme specification.

**Positioning, and it matters.** D1 is a *pre-deployment protocol audit*, not
runtime detection. It runs on the spec alone, returns in well under a second with
no simulation, and fires on a vulnerable scheme whether or not anyone has attacked
it. D2-D5 answer a different question -- "is this run under attack?" -- and the
problem statement needs both. Claiming D1 as runtime detection invites a correct
objection and is not done anywhere in this project.

Four sub-checks, all exact, all from Choi and Kim:

``D1a`` Choi-class Pauli witnesses (Choi section III B; ``docs/derivations.md`` 5.3).
``D1b`` Kim forgeable-message criterion (arXiv:1708.05111; sections 5.3b-5.3c).
``D1c`` positional binding (Choi section III C; section 5.4).
``D1d`` equality-test symmetry (Choi section III C; section 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from pramana.detectors.base import DetectorKind, Verdict
from pramana.engines.base import RoundResults
from pramana.spec.schema import EqualityTest, MessageType, SchemeSpec

REPAIR_ASSISTANT = (
    "Introduce an assistant operation so the encryption set loses its universal "
    "Pauli commutant. Choi section III B proposes (I,H)-type encryption; note that "
    "H alone leaves Q = Y (ledger V-18). A fixed-point-free factor such as S*H "
    "removes every Pauli witness, and Kim Corollary 5's assistant unitary with "
    "three or more rotations removes forgeable messages entirely."
)


@dataclass
class D1Algebra:
    """Static audit. Takes ``results=None`` and never simulates."""

    id: str = "D1"
    kind: DetectorKind = DetectorKind.STATIC

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None = None) -> Verdict:
        """Audit the specification. ``results`` is ignored by design."""
        findings: list[str] = []
        evidence: dict[str, object] = {}

        # --- D1a: Choi-class Pauli witnesses ---
        witnesses = spec.forging_witnesses()
        evidence["d1a_pauli_witnesses"] = list(witnesses)
        if witnesses:
            findings.append(
                f"D1a CHOI_CLASS_RECEIVER_FORGERY: operator(s) {', '.join(witnesses)} "
                f"commute up to global phase with every encryption and rotation "
                f"operator, so a receiver can modify the message and keep the "
                f"signature valid (Choi section III A, Eq. 2)."
            )

        # --- D1b: Kim forgeable-message criterion ---
        try:
            kim = spec.kim_verdict()
            evidence["d1b_theorem"] = kim.theorem
            evidence["d1b_forgeable"] = kim.forgeable
            evidence["d1b_rationale"] = kim.rationale
            if kim.forgeable:
                findings.append(f"D1b FORGEABLE_MESSAGE_EXISTS: {kim.rationale}")
        except NotImplementedError as exc:
            evidence["d1b_theorem"] = "not applicable"
            evidence["d1b_rationale"] = str(exc)

        # --- D1c: positional binding ---
        evidence["d1c_positional_binding"] = spec.binding.positional
        if not spec.binding.positional:
            findings.append(
                "D1c PERMUTATION_MALLEABLE: the scheme operates qubit-wise without "
                "binding position into the signature, so an attacker can reorder "
                "message and signature qubits (Choi section III C). Official and "
                "financial documents have predictable structure, so reordering dates "
                "and amounts is a real attack even without reading the content."
            )

        # --- D1d: equality-test symmetry ---
        evidence["d1d_equality_test"] = spec.verification.equality_test.value
        if spec.verification.equality_test is EqualityTest.SWAP:
            findings.append(
                "D1d DISAVOWAL_VIA_SYMMETRIC_STATE: verification uses a swap test, and "
                "any symmetric state passes it, which the signer can exploit to "
                "disavow a signature (Choi section III C)."
            )

        # --- Scope note the report must carry ---
        if spec.message_type is MessageType.CLASSICAL and not witnesses:
            evidence["scope_note"] = (
                "No Pauli witness exists, but this scheme signs a CLASSICAL message, "
                "and classical verification reads only the Z observable. The Choi "
                "attack then needs only a net operator that flips the computational "
                "basis, which is strictly weaker than commuting up to global phase. "
                "Zero witnesses does not imply this scheme is unforgeable (V-42)."
            )

        fired = bool(findings)
        if fired:
            explanation = (
                f"Static audit of {spec.name!r} found {len(findings)} issue(s), before "
                "any simulation: " + " ".join(findings)
            )
        else:
            explanation = (
                f"Static audit of {spec.name!r} found no Choi-class Pauli witness, no "
                "forgeable message under Kim's criterion, positional binding present, "
                "and no swap-test disavowal path. This clears the audited classes "
                "only; it is not a proof of security."
            )

        return Verdict(
            detector=self.id,
            kind=self.kind,
            fired=fired,
            applicable=True,
            confidence=1.0,  # exact: algebra, not statistics
            evidence=evidence,
            explanation=explanation,
            suggested_fix=REPAIR_ASSISTANT if witnesses else "",
        )
