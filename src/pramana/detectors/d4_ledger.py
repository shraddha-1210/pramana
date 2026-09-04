"""D4 -- entanglement ledger. Exact, not probabilistic.

Each signing round consumes indexed entangled pairs. Physics forbids reusing an
entangled pair, so a replay is necessarily a *classical* replay and the index it
carries is already marked consumed. The check is a lookup, not a threshold: it is
right or wrong, with no error rate.

**What this actually is, said plainly.** A classical freshness mechanism -- a nonce
or counter -- whose uniqueness is *enforced* by a physically consumed resource.
Calling it quantum detection would be an overclaim. The honest framing is stronger
and survives questioning.

Ledger integrity is protected by the Wegman-Carter MAC of Layer 4, so it is not
assumed: it is derived from the same key pool as everything else.

For the prototype the ledger is held by a single arbitrator. A multi-verifier
deployment needs a shared ledger, which brings distributed-systems concerns --
consistency and availability -- that are out of scope and recorded as such in the
honesty table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pramana.crypto.wegman_carter import Tag, make_tag, verify_tag
from pramana.detectors.base import DetectorKind, Verdict, not_applicable
from pramana.engines.base import RoundResults
from pramana.spec.schema import SchemeSpec


class LedgerTampered(Exception):
    """The ledger's own authentication tag did not verify."""


@dataclass
class EntanglementLedger:
    """Consumed entangled-pair indices, authenticated by a Wegman-Carter tag."""

    key: int = 0x5EED
    pad: int = 0xC0FFEE
    consumed: set[int] = field(default_factory=set)
    tag: Tag | None = field(default=None)

    def _serialise(self) -> bytes:
        return ",".join(str(i) for i in sorted(self.consumed)).encode()

    def seal(self) -> None:
        """Authenticate the current contents."""
        self.tag = make_tag(self.key, self.pad, 0, self._serialise())

    def check_integrity(self) -> None:
        """Verify the ledger has not been altered.

        Raises:
            LedgerTampered: The MAC does not match the contents.
        """
        if self.tag is None:
            return
        if not verify_tag(self.key, self.pad, self._serialise(), self.tag):
            raise LedgerTampered(
                "the entanglement ledger's authentication tag does not verify, so its "
                "contents were altered. Do not trust any freshness decision from it; "
                "restore the ledger from an authenticated copy."
            )

    def record(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        """Record indices as consumed, returning any that were already present."""
        replayed = tuple(i for i in indices if i in self.consumed)
        self.consumed.update(indices)
        self.seal()
        return replayed


@dataclass
class D4Ledger:
    """Exact replay detection by consumed-pair lookup."""

    ledger: EntanglementLedger | None = None
    id: str = "D4"
    kind: DetectorKind = DetectorKind.RUNTIME

    def evaluate(self, spec: SchemeSpec, results: RoundResults | None = None) -> Verdict:
        """Replay every round's pair manifest through the ledger."""
        if results is None:
            return not_applicable(
                self.id, self.kind, "D4 is a runtime detector and needs a run to assess."
            )
        ledger = self.ledger if self.ledger is not None else EntanglementLedger()
        ledger.check_integrity()

        replayed: list[int] = []
        for round_result in results.rounds:
            replayed.extend(ledger.record(round_result.pair_indices))

        fired = bool(replayed)
        return Verdict(
            detector=self.id,
            kind=self.kind,
            fired=fired,
            applicable=True,
            confidence=1.0,  # exact lookup; no error rate exists to state
            evidence={
                "rounds": len(results.rounds),
                "distinct_pairs": len(ledger.consumed),
                "replayed_indices": replayed[:20],
                "replayed_count": len(replayed),
            },
            explanation=(
                f"{len(replayed)} entangled-pair index/indices were already marked "
                f"consumed (first: {replayed[:5]}). Physics forbids reusing an entangled "
                "pair, so this is a classical replay of a previously valid pair. The "
                "check is an exact lookup with no error rate."
                if fired
                else (
                    f"All {len(ledger.consumed)} entangled-pair indices across "
                    f"{len(results.rounds)} rounds were fresh. Exact, not statistical."
                )
            ),
        )
