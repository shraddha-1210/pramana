"""The end-to-end demonstration. If this passes, the project's claim holds.

Everything else is presentation.
"""

from __future__ import annotations

import json

import pytest

from pramana.attacks.choi_2011 import Choi2011
from pramana.bridge.assurance_report import AssuranceLevel, build_report
from pramana.bridge.envelope import HybridEnvelope, QuantumAttestation
from pramana.crypto.key_pool import KeyPool
from pramana.decision.table import AttackClass, classify
from pramana.detectors.d1_algebra import D1Algebra
from pramana.detectors.d2_probe_sprt import D2ProbeSprt
from pramana.detectors.d4_ledger import D4Ledger
from pramana.protocol.signing import run_protocol_rounds
from pramana.spec.loader import load_example

PROBE_KEY = b"integration-key0"


@pytest.mark.integration
def test_full_pipeline_choi_scenario() -> None:
    """End-to-end demonstration of the project's thesis.

    1. Load a scheme with a universal Pauli commutant.
    2. D1 flags it statically, before any simulation.
    3. Run Choi's forgery attack against it.
    4. Verification ACCEPTS the forgery, and D2 correctly does not fire.
    5. The decision table classifies from D1's static verdict, with the runtime
       axis reporting a clean run.
    6. The assurance report recommends the fix.
    7. Load the hardened scheme; D1 does not fire.
    """
    # 1. A scheme whose encryption set has a universal Pauli commutant.
    spec = load_example("baseline_bell_aqs")

    # 2. Static audit fires with no simulation at all.
    d1 = D1Algebra().evaluate(spec, None)
    assert d1.fired is True
    assert d1.evidence["d1a_pauli_witnesses"] == ["X", "Y", "Z"]

    # 3-4. The forgery verifies, exactly and always.
    attack = Choi2011().run(spec, trials=100, seed=7)
    assert attack.verification_accepted is True
    assert attack.mismatch_rate == 0.0
    assert attack.message_was_modified is True
    assert attack.attacker_used_key is False
    assert attack.success_rate == 1.0

    honest = run_protocol_rounds(spec, rounds=300, seed=7, probe_key=PROBE_KEY)
    d2 = D2ProbeSprt().evaluate(spec, honest)
    assert d2.fired is False, "the forgery leaves no statistical trace for D2 to find"

    d4 = D4Ledger().evaluate(spec, honest)
    assert d4.fired is False

    # 5. Two axes: unsound scheme, clean run. Both true, and both said.
    runtime_firing = {v.detector for v in (d2, d4) if v.fired}
    result = classify({"D1"} if d1.fired else set(), runtime_firing)
    assert result.static_class is AttackClass.SCHEME_VULNERABLE_TO_RECEIVER_FORGERY
    assert result.runtime_class is AttackClass.NONE
    assert result.is_clean is False

    # 6. The report recommends the repair and fails the audit.
    pool = KeyPool(seed_bits=100_000, bits_per_pair=1)
    pool.generate_from_pairs(2_000)
    pool.record_signing_pairs(8_000)
    for _ in range(300):
        pool.record_signature()
    pool.consume(300 * 127, "one pad per signature")

    report = build_report(
        spec=spec,
        run_id="demo-0001",
        seed=7,
        verdicts=[d1, d2, d4],
        classification=result,
        key_budget=pool.budget(),
        attack_results=[{"attack": attack.attack, "success_rate": attack.success_rate}],
    )
    payload = json.loads(report.to_json())

    assert payload["assurance_level"] == AssuranceLevel.L0_FAILED.value
    assert "assistant" in payload["verdicts"][0]["suggested_fix"]
    assert payload["honesty_table"]["out_of_scope"], "every report carries the honesty table"
    assert any(a["detector"] == "D1" for a in payload["incident_response"])
    assert payload["key_budget"]["summary"]

    # 7. The scheme that clears the audit.
    hardened = load_example("kim_forgery_free_aqs")
    assert D1Algebra().evaluate(hardened, None).fired is False


@pytest.mark.integration
def test_choi_fixed_still_fails_and_the_report_says_why() -> None:
    """Applying the published fix is not enough, and the tool says so."""
    spec = load_example("choi_fixed_aqs")

    d1 = D1Algebra().evaluate(spec, None)
    assert d1.fired is True
    assert d1.evidence["d1a_pauli_witnesses"] == ["Y"]

    attack = Choi2011(forge="Y").run(spec, trials=50, seed=3)
    assert attack.success_rate == 1.0

    report = build_report(spec, "demo-0002", 3, [d1])
    assert report.assurance_level() is AssuranceLevel.L0_FAILED


@pytest.mark.integration
def test_report_json_round_trips_and_always_carries_the_honesty_table() -> None:
    """A report that omits its own limits is not an assurance report."""
    spec = load_example("baseline_bell_aqs")
    report = build_report(spec, "r1", 1, [D1Algebra().evaluate(spec)])
    payload = json.loads(report.to_json())

    for section in ("simulated", "needs_real_hardware", "out_of_scope", "known_limits"):
        assert payload["honesty_table"][section]
    assert payload["forgery_bounds"]["choi_forgery_success"].startswith("1.0")


@pytest.mark.integration
def test_envelope_round_trips_with_all_three_paths() -> None:
    """Build -> serialise -> parse, with the tiers labelled separately."""
    envelope = HybridEnvelope(
        classical_signature="classical-sig",
        pqc_signature="pqc-sig",
        pqc_mechanism="ML-DSA-65",
        attestation=QuantumAttestation(
            scheme="baseline_bell_aqs",
            run_id="r1",
            seed=7,
            accepted=True,
            detector_summary={"D1": True, "D2": False},
        ),
    )
    parsed = HybridEnvelope.parse(envelope.to_json())

    assert parsed == envelope
    tiers = parsed.security_tiers()
    assert tiers["quantum"].startswith("information-theoretic")
    assert "quantum-vulnerable" in tiers["classical"]
    assert "PROPOSED EXTENSION" in parsed.standards_note()


@pytest.mark.integration
def test_envelope_rejects_an_unknown_version() -> None:
    with pytest.raises(ValueError, match="unknown envelope version"):
        HybridEnvelope.parse(json.dumps({"version": "something-else"}))


@pytest.mark.integration
def test_determinism_across_the_whole_pipeline() -> None:
    """Same seed, same report. Rule 6, end to end."""
    spec = load_example("baseline_bell_aqs")

    def once() -> str:
        run = run_protocol_rounds(spec, rounds=100, seed=99, probe_key=PROBE_KEY)
        verdicts = [D1Algebra().evaluate(spec), D2ProbeSprt().evaluate(spec, run)]
        return build_report(spec, "fixed-id", 99, verdicts).to_json()

    assert once() == once()
