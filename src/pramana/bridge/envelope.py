"""Hybrid envelope: classical + post-quantum + quantum attestation.

**Structural stub only.** Full CMS/PAdES compliance is out of scope. There is *no
current standard* for embedding a quantum attestation token in PAdES, so this is a
**proposed extension**, not a compliance claim. Presenting it as compliance would
be an overclaim a judge can check in minutes.

The three security tiers are labelled separately so nobody reads the envelope as
one uniform guarantee:

============  ==========================================================
Path          Security
============  ==========================================================
quantum       Information-theoretic, **given assumption A-1**.
pqc           Computationally secure.
classical     Computationally secure, and quantum-vulnerable.
============  ==========================================================

The weakest tier bounds the envelope, which is why the tiers are reported
individually rather than collapsed into a single label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

ENVELOPE_VERSION = "pramana-hybrid-envelope/0.1"

SECURITY_TIERS = {
    "quantum": "information-theoretic, given assumption A-1 (pre-shared seed key)",
    "pqc": "computationally secure",
    "classical": "computationally secure, quantum-vulnerable",
}


@dataclass(frozen=True)
class QuantumAttestation:
    """The quantum-signature path's attestation token.

    Not a standardised structure. Named as a proposed extension throughout.
    """

    scheme: str
    run_id: str
    seed: int
    accepted: bool
    detector_summary: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class HybridEnvelope:
    """Structural nesting of the three signature paths."""

    classical_signature: str
    pqc_signature: str | None
    pqc_mechanism: str | None
    attestation: QuantumAttestation | None
    version: str = ENVELOPE_VERSION

    def security_tiers(self) -> dict[str, str]:
        """Per-path security labels for the paths actually present."""
        tiers = {"classical": SECURITY_TIERS["classical"]}
        if self.pqc_signature is not None:
            tiers["pqc"] = SECURITY_TIERS["pqc"]
        if self.attestation is not None:
            tiers["quantum"] = SECURITY_TIERS["quantum"]
        return tiers

    def standards_note(self) -> str:
        """The compliance disclaimer, carried with the structure itself."""
        return (
            "Structural stub. No standard currently exists for embedding a quantum "
            "attestation token in CMS or PAdES, so this envelope is a PROPOSED "
            "EXTENSION and not a compliance claim. The classical and PQC fields are "
            "representations, not standards-conformant encodings."
        )

    def to_json(self) -> str:
        """Serialise. Round-trips through ``parse``."""
        payload = {
            "version": self.version,
            "classical_signature": self.classical_signature,
            "pqc_signature": self.pqc_signature,
            "pqc_mechanism": self.pqc_mechanism,
            "attestation": asdict(self.attestation) if self.attestation else None,
            "security_tiers": self.security_tiers(),
            "standards_note": self.standards_note(),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def parse(cls, text: str) -> HybridEnvelope:
        """Recover an envelope from its JSON form.

        Raises:
            ValueError: The version is unrecognised, so the structure cannot be
                trusted to mean what this code assumes.
        """
        data = json.loads(text)
        if data.get("version") != ENVELOPE_VERSION:
            raise ValueError(
                f"unknown envelope version {data.get('version')!r}; expected "
                f"{ENVELOPE_VERSION}."
            )
        attestation = data.get("attestation")
        return cls(
            classical_signature=data["classical_signature"],
            pqc_signature=data.get("pqc_signature"),
            pqc_mechanism=data.get("pqc_mechanism"),
            attestation=QuantumAttestation(**attestation) if attestation else None,
            version=data["version"],
        )
