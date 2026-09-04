"""Layer 8: PQC adapter, hybrid envelope, assurance report, incident response."""

from __future__ import annotations

import json

import pytest

from pramana.bridge.assurance_report import HONESTY_TABLE, AssuranceLevel, build_report
from pramana.bridge.envelope import ENVELOPE_VERSION, HybridEnvelope, QuantumAttestation
from pramana.bridge.incident_response import INCIDENT_RESPONSE, action_for
from pramana.bridge.pqc_adapter import PQC_ENV_FLAG, pqc_status
from pramana.detectors.d1_algebra import D1Algebra
from pramana.spec.loader import load_example

DETECTOR_IDS = ("D1", "D2", "D3", "D4", "D5")


# --------------------------------------------------------------------------
# Incident response.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("detector", DETECTOR_IDS)
def test_every_detector_has_an_incident_response_entry(detector: str) -> None:
    """A detection with no response path is an incomplete framework."""
    entry = action_for(detector)
    assert entry.severity in {"critical", "high", "medium", "low"}
    assert entry.action.strip()
    assert entry.rationale.strip()


def test_unknown_detector_raises_rather_than_returning_a_default() -> None:
    with pytest.raises(KeyError, match="no incident-response entry"):
        action_for("D99")


def test_d1_response_is_a_certification_decision_not_an_incident() -> None:
    """Positioning: D1 means the scheme is unsound, not that a run is under attack."""
    entry = action_for("D1")
    assert "Do not certify" in entry.action
    assert "certification decision" in entry.rationale


# --------------------------------------------------------------------------
# PQC adapter.
# --------------------------------------------------------------------------


def test_pqc_is_off_by_default_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent PQC degrades gracefully; it never crashes the caller."""
    monkeypatch.delenv(PQC_ENV_FLAG, raising=False)
    status = pqc_status()

    assert status.available is False
    assert PQC_ENV_FLAG in status.reason
    assert status.mechanisms == ()


def test_pqc_probe_never_raises_even_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag set but liboqs missing, it reports rather than propagating."""
    monkeypatch.setenv(PQC_ENV_FLAG, "1")
    status = pqc_status()
    assert isinstance(status.available, bool)
    assert status.reason


def test_pqc_round_trip_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skips cleanly with a stated reason when liboqs is absent."""
    monkeypatch.setenv(PQC_ENV_FLAG, "1")
    status = pqc_status()
    if not status.available:
        pytest.skip(f"liboqs unavailable: {status.reason}")

    from pramana.bridge.pqc_adapter import sign, verify

    message = b"assurance report digest"
    signature, public_key, mechanism = sign(message)
    assert verify(message, signature, public_key, mechanism) is True
    assert verify(b"tampered", signature, public_key, mechanism) is False


def test_pqc_mechanism_names_are_never_hardcoded() -> None:
    """Rule 4: the adapter enumerates, it does not assume."""
    import inspect

    from pramana.bridge import pqc_adapter

    source = inspect.getsource(pqc_adapter)
    assert "get_enabled_sig_mechanisms" in source


# --------------------------------------------------------------------------
# Envelope.
# --------------------------------------------------------------------------


def test_envelope_round_trips_all_three_components() -> None:
    envelope = HybridEnvelope(
        classical_signature="c",
        pqc_signature="p",
        pqc_mechanism="ML-DSA-44",
        attestation=QuantumAttestation("s", "r", 1, True, {"D1": True}),
    )
    assert HybridEnvelope.parse(envelope.to_json()) == envelope


def test_envelope_labels_the_three_security_tiers_separately() -> None:
    """Collapsing them into one label would misdescribe the weakest path."""
    envelope = HybridEnvelope("c", "p", "ML-DSA-44", QuantumAttestation("s", "r", 1, True))
    tiers = envelope.security_tiers()

    assert tiers["quantum"].startswith("information-theoretic")
    assert "A-1" in tiers["quantum"], "the assumption must travel with the claim"
    assert tiers["pqc"] == "computationally secure"
    assert "quantum-vulnerable" in tiers["classical"]


def test_envelope_omits_tiers_for_absent_paths() -> None:
    envelope = HybridEnvelope("c", None, None, None)
    assert set(envelope.security_tiers()) == {"classical"}


def test_envelope_carries_its_own_standards_disclaimer() -> None:
    """No standard exists for this. The structure says so itself."""
    note = HybridEnvelope("c", None, None, None).standards_note()
    assert "PROPOSED EXTENSION" in note
    assert "not a compliance claim" in note


def test_envelope_version_is_pinned() -> None:
    assert HybridEnvelope("c", None, None, None).version == ENVELOPE_VERSION


# --------------------------------------------------------------------------
# Assurance report.
# --------------------------------------------------------------------------


def test_report_is_valid_json_with_the_expected_sections() -> None:
    spec = load_example("baseline_bell_aqs")
    payload = json.loads(build_report(spec, "r", 1, [D1Algebra().evaluate(spec)]).to_json())

    for key in (
        "version",
        "run_id",
        "seed",
        "scheme",
        "assurance_level",
        "verdicts",
        "forgery_bounds",
        "incident_response",
        "honesty_table",
    ):
        assert key in payload, key


def test_report_always_contains_the_honesty_table() -> None:
    """Non-negotiable: the report states its own limits."""
    spec = load_example("kim_forgery_free_aqs")
    payload = json.loads(build_report(spec, "r", 1, []).to_json())

    assert payload["honesty_table"] == HONESTY_TABLE
    assert any("Z observable" in limit for limit in HONESTY_TABLE["known_limits"])
    assert any("not analytically derived" in limit for limit in HONESTY_TABLE["known_limits"])


def test_a_fired_detector_fails_the_audit() -> None:
    spec = load_example("baseline_bell_aqs")
    report = build_report(spec, "r", 1, [D1Algebra().evaluate(spec)])
    assert report.assurance_level() is AssuranceLevel.L0_FAILED


def test_a_clean_scheme_reaches_only_the_simulated_tier() -> None:
    """Simulation cannot reach L3 or L4. Hardware characterisation is out of scope."""
    spec = load_example("kim_forgery_free_aqs")
    report = build_report(spec, "r", 1, [D1Algebra().evaluate(spec)])

    assert report.assurance_level() is AssuranceLevel.L1_SIMULATED
    assert "not reachable by simulation" in AssuranceLevel.L3_HARDWARE.value
    assert "not reachable by simulation" in AssuranceLevel.L4_CERTIFIED.value


def test_forgery_bounds_each_name_their_derivation() -> None:
    """A bound without its derivation is a magic number."""
    spec = load_example("baseline_bell_aqs")
    bounds = build_report(spec, "r", 1, []).forgery_bounds()

    assert bounds["blind_forgery_correct_guess"].startswith("4**-n")
    assert bounds["blind_forgery_acceptance"].startswith("2**-n")
    assert "six-state" in bounds["intercept_resend_escape"]
    assert bounds["choi_forgery_success"].startswith("1.0 exactly")
    for text in bounds.values():
        assert "(" in text, f"bound {text!r} does not state its derivation"


def test_incident_actions_are_emitted_only_for_detectors_that_fired() -> None:
    spec = load_example("baseline_bell_aqs")
    fired = build_report(spec, "r", 1, [D1Algebra().evaluate(spec)]).incident_actions()
    assert [a["detector"] for a in fired] == ["D1"]

    clean_spec = load_example("kim_forgery_free_aqs")
    clean = build_report(clean_spec, "r", 1, [D1Algebra().evaluate(clean_spec)])
    assert clean.incident_actions() == []


def test_every_incident_entry_is_reachable_from_a_detector_id() -> None:
    assert set(INCIDENT_RESPONSE) == set(DETECTOR_IDS)
