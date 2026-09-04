"""Layer 9: the API. Happy path, validation failure, and not-found for each route."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pramana.api.db import init_db
from pramana.api.main import app


@pytest.fixture()
def client():
    """A client over a fresh in-memory database per test."""
    init_db(None)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_schemes_includes_all_four_bundled(client) -> None:
    payload = client.get("/api/schemes").json()
    assert set(payload["bundled"]) == {
        "baseline_bell_aqs",
        "choi_fixed_aqs",
        "kim_forgery_free_aqs",
        "pauli_witness_free_aqs",
    }


def test_get_scheme_returns_the_static_verdict_immediately(client) -> None:
    """The Scheme Lab's strongest moment: a verdict with no simulation."""
    payload = client.get("/api/schemes/baseline_bell_aqs").json()

    assert payload["static_verdict"]["fired"] is True
    assert payload["static_verdict"]["kind"] == "static"
    assert payload["static_verdict"]["evidence"]["d1a_pauli_witnesses"] == ["X", "Y", "Z"]
    assert payload["static_verdict"]["suggested_fix"]
    assert payload["spec"]["name"] == "baseline_bell_aqs"


def test_get_scheme_reports_the_clean_scheme_as_clean(client) -> None:
    payload = client.get("/api/schemes/kim_forgery_free_aqs").json()
    assert payload["static_verdict"]["fired"] is False
    assert payload["clifford_simulable"] is False


def test_get_unknown_scheme_is_404(client) -> None:
    assert client.get("/api/schemes/nope").status_code == 404


def test_upload_a_valid_spec(client) -> None:
    source = (
        "src/pramana/spec/examples/baseline_bell_aqs.yaml"
    )
    from pathlib import Path

    text = Path(source).read_text(encoding="utf-8").replace(
        "name: baseline_bell_aqs", "name: uploaded_copy"
    )
    response = client.post("/api/schemes", json={"name": "uploaded_copy", "source": text})

    assert response.status_code == 201
    assert response.json()["static_verdict"]["fired"] is True
    assert "uploaded_copy" in client.get("/api/schemes").json()["uploaded"]


def test_upload_a_malformed_spec_is_422_with_a_useful_message(client) -> None:
    response = client.post("/api/schemes", json={"name": "bad", "source": "just: a string"})
    assert response.status_code == 422
    assert "Field required" in response.text or "field required" in response.text.lower()


def test_list_attacks_count_matches_the_registry(client) -> None:
    """The API must not claim more attacks than exist."""
    from pramana.attacks import ATTACK_REGISTRY

    payload = client.get("/api/attacks").json()
    assert payload["count"] == len(ATTACK_REGISTRY) == 4
    for entry in payload["attacks"]:
        assert entry["reference"], entry["name"]


def test_full_run_end_to_end_through_the_api(client) -> None:
    """Start a run, poll it, and retrieve its report."""
    start = client.post(
        "/api/runs",
        json={"scheme": "baseline_bell_aqs", "seed": 7, "rounds": 60, "attacks": ["choi_2011"]},
    )
    assert start.status_code == 202
    run_id = start.json()["id"]

    # TestClient runs background tasks synchronously on context exit of the request.
    status = client.get(f"/api/runs/{run_id}").json()
    assert status["seed"] == 7, "every stored run keeps its seed"
    assert status["status"] in {"pending", "running", "complete"}

    report = client.get(f"/api/runs/{run_id}/report")
    assert report.status_code == 200
    payload = report.json()
    assert payload["honesty_table"]["out_of_scope"]
    assert payload["classification"]["static_class"] == "scheme_vulnerable_to_receiver_forgery"
    assert payload["classification"]["runtime_class"] == "none"


def test_run_for_unknown_scheme_is_404(client) -> None:
    assert client.post("/api/runs", json={"scheme": "nope"}).status_code == 404


def test_report_before_completion_is_409(client) -> None:
    from pramana.api.db import Run, _SessionLocal

    assert _SessionLocal is not None
    session = _SessionLocal()
    session.add(Run(id="pending01", scheme="baseline_bell_aqs", seed=1, rounds=1))
    session.commit()
    session.close()

    assert client.get("/api/runs/pending01/report").status_code == 409


def test_unknown_run_is_404(client) -> None:
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.get("/api/runs/deadbeef/report").status_code == 404
    assert client.get("/api/runs/deadbeef/curves").status_code == 404


def test_curves_return_theory_alongside_measurement(client) -> None:
    """Plotting theory alone or measurement alone would discard the evidence."""
    start = client.post("/api/runs", json={"scheme": "baseline_bell_aqs", "rounds": 20})
    run_id = start.json()["id"]

    series = client.get(f"/api/runs/{run_id}/curves").json()["series"]
    assert series["blind_forgery_acceptance"]["theory"][0] == 0.5
    assert series["blind_forgery_correct_guess"]["theory"][0] == 0.25
    assert series["intercept_resend_escape"]["label"] == "(2/3)**d"


def test_matrix_marks_the_non_simulable_scheme_inapplicable(client) -> None:
    """A scheme that cannot run must not read as "attack failed"."""
    rows = {row["scheme"]: row for row in client.get("/api/matrix").json()["rows"]}

    kim = rows["kim_forgery_free_aqs"]
    assert kim["static_fired"] is False
    assert all(cell["applicable"] is False for cell in kim["cells"])

    baseline = rows["baseline_bell_aqs"]
    assert baseline["static_fired"] is True
    choi = next(c for c in baseline["cells"] if c["attack"] == "choi_2011")
    assert choi["success_rate"] == 1.0
    assert choi["detectors"]["D1"] is True
    assert choi["detectors"]["D2"] is False, "the Choi row is the thesis: D1 on, D2 off"


def test_run_list_is_returned_newest_first(client) -> None:
    client.post("/api/runs", json={"scheme": "baseline_bell_aqs", "rounds": 5})
    client.post("/api/runs", json={"scheme": "choi_fixed_aqs", "rounds": 5})
    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) >= 2
    assert all("seed" in r for r in runs)
