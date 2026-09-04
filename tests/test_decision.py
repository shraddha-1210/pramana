"""Layer 7: the decision table. Deterministic, total, two-axis."""

from __future__ import annotations

import pytest

from pramana.attacks import ATTACK_REGISTRY
from pramana.decision.table import (
    RUNTIME_DECISION_TABLE,
    STATIC_DECISION_TABLE,
    AttackClass,
    classify,
)


def test_honest_traffic_on_a_sound_scheme_is_clean() -> None:
    """Both axes clear. The only state meaning "nothing to report"."""
    result = classify(static_firing=set(), runtime_firing=set())
    assert result.static_class is AttackClass.NONE
    assert result.runtime_class is AttackClass.NONE
    assert result.is_clean is True


def test_honest_traffic_on_a_vulnerable_scheme_separates_the_axes() -> None:
    """Ledger Q-5, resolved.

    D1 fires on a vulnerable scheme even under perfectly honest traffic. A
    single-axis table would have to call that an attack on a run where none
    occurred. Two axes let the tool say the true thing: the scheme is unsound and
    this run was clean.
    """
    result = classify(static_firing={"D1"}, runtime_firing=set())

    assert result.static_class is AttackClass.SCHEME_VULNERABLE_TO_RECEIVER_FORGERY
    assert result.runtime_class is AttackClass.NONE
    assert result.is_clean is False
    assert "nothing to see" in result.explanation


@pytest.mark.parametrize(
    ("firing", "expected"),
    [
        ({"D2"}, AttackClass.CHANNEL_OR_OUTSIDER),
        ({"D3"}, AttackClass.REPUDIATION),
        ({"D4"}, AttackClass.REPLAY),
        ({"D5"}, AttackClass.IMPERSONATION),
        ({"D3", "D5"}, AttackClass.MALICIOUS_ARBITRATOR),
        (set(), AttackClass.NONE),
    ],
)
def test_every_runtime_entry_maps_as_specified(firing: set[str], expected: AttackClass) -> None:
    assert classify(set(), firing).runtime_class is expected


def test_unmapped_runtime_combination_is_surfaced_not_swallowed() -> None:
    """An unanticipated pattern is information, not a default case."""
    result = classify(set(), {"D2", "D4"})

    assert result.runtime_class is AttackClass.UNCLASSIFIED
    assert result.runtime_firing == frozenset({"D2", "D4"})
    assert "Investigate rather than dismiss" in result.explanation


def test_the_table_is_total_and_no_input_raises() -> None:
    """Every subset of the detector set classifies without error."""
    from itertools import chain, combinations

    detectors = ["D2", "D3", "D4", "D5"]
    subsets = chain.from_iterable(
        combinations(detectors, r) for r in range(len(detectors) + 1)
    )
    for subset in subsets:
        for static in ((), ("D1",)):
            result = classify(set(static), set(subset))
            assert isinstance(result.static_class, AttackClass)
            assert isinstance(result.runtime_class, AttackClass)
            assert result.explanation.strip()


def test_d1_never_appears_on_the_runtime_axis() -> None:
    """Mixing the axes is the error this table exists to avoid."""
    for key in RUNTIME_DECISION_TABLE:
        assert "D1" not in key
    for key in STATIC_DECISION_TABLE:
        assert key <= {"D1"}


def test_every_table_entry_is_reachable_by_some_firing_pattern() -> None:
    """No dead rows: each mapped pattern is producible."""
    for pattern, expected in RUNTIME_DECISION_TABLE.items():
        assert classify(set(), set(pattern)).runtime_class is expected
    for pattern, expected in STATIC_DECISION_TABLE.items():
        assert classify(set(pattern), set()).static_class is expected


def test_both_axes_can_reach_a_non_none_and_a_none_verdict() -> None:
    """Ledger Q-13: neither axis is stuck.

    Static reaches NONE (kim_forgery_free_aqs) and non-NONE (baseline). Runtime
    reaches NONE (honest traffic) and non-NONE (any firing detector).
    """
    assert classify(set(), set()).static_class is AttackClass.NONE
    assert classify({"D1"}, set()).static_class is not AttackClass.NONE
    assert classify(set(), set()).runtime_class is AttackClass.NONE
    assert classify(set(), {"D4"}).runtime_class is not AttackClass.NONE


def test_every_registered_attack_has_a_threat_class_the_table_can_express() -> None:
    """An attack whose class the table cannot name would be unreportable."""
    expressible = {c.value for c in AttackClass}
    for name, adversary in ATTACK_REGISTRY.items():
        assert adversary.threat_class, name
        # Threat classes are attack-side labels; each must correspond to a
        # decision-table outcome the report can render.
        assert any(
            adversary.threat_class in value or value in adversary.threat_class
            for value in expressible
        ) or adversary.threat_class in {
            "receiver_forgery",
            "outsider_forgery",
            "channel_manipulation",
            "replay",
        }, name
