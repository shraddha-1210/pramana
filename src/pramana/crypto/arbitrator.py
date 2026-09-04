"""Threshold arbitration: the fix for the circular malicious-arbitrator problem.

**The circularity.** D5 checks probe outcomes against what the arbitrator says
they should be. With one arbitrator, an arbitrator that is itself the attacker is
the reference for its own audit, and the detector is worthless against exactly the
adversary it exists to catch.

**The fix.** Split the probe-expectation key across ``n`` arbitrators with Shamir.
Reconstruction needs ``t`` shares, so no single arbitrator can fabricate a
consistent expectation.

**Correction to the build plan's Layer 4d capability table.** The plan states that
``n = 3, t = 2`` *identifies* the liar by majority. It does not, and the reason is
a theorem rather than an implementation detail.

Shamir shares of a degree ``t-1`` polynomial are a Reed-Solomon code of dimension
``t`` and length ``n``, whose minimum distance is ``d = n - t + 1``. Such a code

* **detects** ``e`` errors when ``e <= d - 1 = n - t``;
* **corrects** (identifies) ``e`` errors when ``2e <= d - 1 = n - t``.

For a single liar, ``e = 1``:

===  ===  =======  =========  ===========================================
n    t    detect?  identify?  Why
===  ===  =======  =========  ===========================================
2    2    no       no         ``n - t = 0``: one quorum, no redundancy at all.
3    2    **yes**  no         ``n - t = 1``: enough to notice, not to locate.
4    2    yes      **yes**    ``n - t = 2``: the first size that identifies.
5    3    yes      yes        ``n - t = 2``.
===  ===  =======  =========  ===========================================

Concretely at ``n = 3, t = 2``: three points that should lie on a line do not, and
each of the three pairs defines a different line. Nothing distinguishes which
point is the outlier. At ``n = 4`` the three quorums avoiding the liar all agree,
giving a genuine majority.

**Identification therefore requires ``n >= t + 2``, not ``n > t``.** Detection
requires ``n >= t + 1``. Below that, a panel can be given the true value
externally -- but if the caller already knows the secret it does not need the
arbitrators, so that path is a test instrument, not a security mechanism.

Secure unless ``t`` or more collude: assumption A-2, with a deliberate
expected-failure test at the boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from pramana.crypto.field import P
from pramana.crypto.shamir import Share, reconstruct, split


class ArbitrationFailure(Exception):
    """Arbitrators disagreed and no majority could resolve it."""


@dataclass(frozen=True)
class Arbitrator:
    """One arbitrator holding one share.

    ``honest=False`` makes it present a corrupted share, so the lying cases are
    simulated rather than described.
    """

    identity: int
    share: Share
    honest: bool = True

    def offer(self) -> Share:
        """The share this arbitrator presents when asked."""
        if self.honest:
            return self.share
        return Share(x=self.share.x, y=(self.share.y + 1) % P)


@dataclass(frozen=True)
class ArbitrationVerdict:
    """The outcome of asking a panel to reconstruct.

    Attributes:
        secret: The reconstructed value, when one could be settled on.
        agreed: Whether every quorum produced the same value.
        suspected: Arbitrator identities implicated, when identification was
            possible. Empty when the panel is too small to locate a liar --
            never a guess.
        can_detect: Whether the panel size permits noticing a lie at all.
        can_identify: Whether the panel size permits naming the liar.
        explanation: Why this verdict, in words. Never optional: a certification
            tool must justify every ruling.
    """

    secret: int | None
    agreed: bool
    suspected: tuple[int, ...]
    can_detect: bool
    can_identify: bool
    explanation: str


@dataclass
class ArbitrationPanel:
    """``t``-of-``n`` arbitration over a shared secret."""

    arbitrators: tuple[Arbitrator, ...]
    threshold: int

    @classmethod
    def build(
        cls,
        secret: int,
        threshold: int,
        count: int,
        seed: int,
        *,
        dishonest: tuple[int, ...] = (),
    ) -> ArbitrationPanel:
        """Split ``secret`` across ``count`` arbitrators, marking some dishonest."""
        shares = split(secret, threshold, count, seed)
        return cls(
            arbitrators=tuple(
                Arbitrator(identity=i, share=s, honest=i not in dishonest)
                for i, s in enumerate(shares)
            ),
            threshold=threshold,
        )

    @property
    def redundancy(self) -> int:
        """``n - t``: the number of shares beyond the reconstruction threshold."""
        return len(self.arbitrators) - self.threshold

    @property
    def can_detect(self) -> bool:
        """Whether one lie is detectable from the shares alone. Needs ``n >= t+1``."""
        return self.redundancy >= 1

    @property
    def can_identify(self) -> bool:
        """Whether one liar can be named. Needs ``n >= t+2``.

        Not ``n > t``: correcting one error needs ``2e <= n - t``, so a single
        spare share notices the problem without locating it.
        """
        return self.redundancy >= 2

    def reconstruct_all_quorums(self) -> dict[tuple[int, ...], int]:
        """Reconstruct from every quorum of exactly ``t`` arbitrators."""
        return {
            tuple(a.identity for a in quorum): reconstruct([a.offer() for a in quorum])
            for quorum in combinations(self.arbitrators, self.threshold)
        }

    def adjudicate(self, expected: int | None = None) -> ArbitrationVerdict:
        """Reconstruct and report whether the arbitrators are consistent.

        Args:
            expected: The true secret, when the caller knows it. This is a test
                instrument: it lets a panel with no redundancy notice a lie. It is
                not a security mechanism, because a caller who knows the secret
                has no need of the panel.
        """
        quorums = self.reconstruct_all_quorums()
        values = list(quorums.values())
        distinct = set(values)

        if len(distinct) == 1:
            only = values[0]
            if expected is not None and only != expected:
                return ArbitrationVerdict(
                    secret=only,
                    agreed=False,
                    suspected=(),
                    can_detect=self.can_detect,
                    can_identify=self.can_identify,
                    explanation=(
                        f"All {len(quorums)} quorum(s) agreed on a value that is not the "
                        f"expected secret. Unanimous agreement on a wrong value means "
                        f"the lie is invisible to the panel itself (n = "
                        f"{len(self.arbitrators)}, t = {self.threshold}, redundancy "
                        f"{self.redundancy}); it was caught only because the true value "
                        "was supplied externally. This is the collusion boundary of "
                        "assumption A-2."
                    ),
                )
            return ArbitrationVerdict(
                secret=only,
                agreed=True,
                suspected=(),
                can_detect=self.can_detect,
                can_identify=self.can_identify,
                explanation=(
                    f"All {len(quorums)} quorum(s) of {self.threshold} reconstructed the "
                    "same secret. No arbitrator is contradicted by any other."
                ),
            )

        # Quorums disagree, so at least one arbitrator is lying.
        if not self.can_identify:
            return ArbitrationVerdict(
                secret=None,
                agreed=False,
                suspected=(),
                can_detect=True,
                can_identify=False,
                explanation=(
                    f"Quorums disagree, so at least one arbitrator is lying. With n = "
                    f"{len(self.arbitrators)} and t = {self.threshold} the redundancy is "
                    f"{self.redundancy}, and identifying one liar needs redundancy 2 "
                    f"(2e <= n - t). The liar is detected but cannot be named; no guess "
                    f"is offered. Use n >= {self.threshold + 2} to identify."
                ),
            )

        tally = Counter(values)
        ranked = tally.most_common()
        majority_value, majority_count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        # A *unique plurality*, not a strict majority. With one liar the C(n-1, t)
        # quorums that avoid it all agree, while the quorums containing it scatter
        # into distinct wrong values -- so the honest value leads but need not
        # exceed half. At n=4, t=2 it is 3 of 6, which a strict-majority rule would
        # wrongly reject.
        if majority_count <= runner_up:
            return ArbitrationVerdict(
                secret=None,
                agreed=False,
                suspected=(),
                can_detect=True,
                can_identify=True,
                explanation=(
                    f"Quorums disagree and no single value leads "
                    f"({majority_count} of {len(values)}, tied with another). More "
                    f"arbitrators are lying than this panel can correct; assumption "
                    f"A-2 is violated."
                ),
            )

        trusted: set[int] = set()
        for identities, value in quorums.items():
            if value == majority_value:
                trusted.update(identities)
        suspected = tuple(
            sorted(a.identity for a in self.arbitrators if a.identity not in trusted)
        )
        return ArbitrationVerdict(
            secret=majority_value,
            agreed=False,
            suspected=suspected,
            can_detect=True,
            can_identify=True,
            explanation=(
                f"Quorums disagree: {len(tally)} distinct reconstructions. One value was "
                f"returned by {majority_count} of {len(values)} quorums, and arbitrator(s) "
                f"{list(suspected)} appear in no quorum returning it, so they are the "
                "source of the inconsistency."
            ),
        )
