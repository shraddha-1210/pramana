"""Post-quantum signature adapter. Optional, import-guarded, degrades gracefully.

**Algorithm names are enumerated at runtime, never hardcoded** (Rule 4). liboqs
changes its enabled mechanism list between builds, so the adapter asks what is
present and selects from that.

**Why the import is guarded behind a flag.** ``import oqs`` is not a passive
import: liboqs-python 0.16.0 runs ``git clone`` of the liboqs C library followed by
a cmake build *at import time* when no shared library is installed. On a machine
without cmake it fails after minutes of network and disk activity. So the import
happens only when ``PRAMANA_ENABLE_PQC=1`` is set (ledger V-14).

liboqs is explicitly not production-ready per its own maintainers. Fine for a
prototype; never claimed as deployment-ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PQC_ENV_FLAG = "PRAMANA_ENABLE_PQC"


@dataclass(frozen=True)
class PqcStatus:
    """Whether the post-quantum path is available, and why not when it is not."""

    available: bool
    reason: str
    mechanisms: tuple[str, ...] = ()
    selected: str | None = None


def pqc_status() -> PqcStatus:
    """Probe the PQC backend without ever crashing the caller."""
    if os.environ.get(PQC_ENV_FLAG) != "1":
        return PqcStatus(
            available=False,
            reason=(
                f"PQC is disabled. Set {PQC_ENV_FLAG}=1 to enable it. The import is "
                "gated because importing oqs triggers a git clone and cmake build of "
                "liboqs at import time (ledger V-14)."
            ),
        )
    try:
        import oqs
    except Exception as exc:  # noqa: BLE001 - report, never propagate
        return PqcStatus(
            available=False,
            reason=f"liboqs unavailable ({type(exc).__name__}: {exc}).",
        )

    mechanisms = tuple(oqs.get_enabled_sig_mechanisms())
    preferred = [m for m in mechanisms if "ML-DSA" in m]
    return PqcStatus(
        available=bool(mechanisms),
        reason="liboqs available.",
        mechanisms=mechanisms,
        selected=preferred[0] if preferred else (mechanisms[0] if mechanisms else None),
    )


def sign(message: bytes) -> tuple[bytes, bytes, str]:
    """Sign with the selected mechanism.

    Returns:
        ``(signature, public_key, mechanism_name)``.

    Raises:
        RuntimeError: The PQC path is unavailable. The caller is expected to fall
            back to the classical and quantum paths, which do not depend on it.
    """
    status = pqc_status()
    if not status.available or status.selected is None:
        raise RuntimeError(f"PQC signing unavailable: {status.reason}")

    import oqs

    with oqs.Signature(status.selected) as signer:
        public_key = signer.generate_keypair()
        return signer.sign(message), public_key, status.selected


def verify(message: bytes, signature: bytes, public_key: bytes, mechanism: str) -> bool:
    """Verify a post-quantum signature.

    Raises:
        RuntimeError: The PQC path is unavailable.
    """
    status = pqc_status()
    if not status.available:
        raise RuntimeError(f"PQC verification unavailable: {status.reason}")

    import oqs

    with oqs.Signature(mechanism) as verifier:
        result: bool = verifier.verify(message, signature, public_key)
        return result
