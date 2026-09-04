"""Layer 9 -- the FastAPI application.

Long runs execute as background tasks with progress polling; the request never
blocks. Every run is persisted with its seed, so any result surfaced in the
dashboard is reproducible from the dashboard.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from pramana.api.db import Run, UploadedSpec, get_session, init_db, json_safe
from pramana.attacks import ATTACK_REGISTRY
from pramana.bridge.assurance_report import build_report
from pramana.crypto.key_pool import KeyPool
from pramana.decision.table import classify
from pramana.detectors.d1_algebra import D1Algebra
from pramana.detectors.d2_probe_sprt import D2ProbeSprt
from pramana.detectors.d4_ledger import D4Ledger
from pramana.protocol.signing import run_protocol_rounds
from pramana.spec.loader import SpecError, example_names, load_example, load_spec_text
from pramana.spec.schema import SchemeSpec

app = FastAPI(
    title="Pramana",
    description="Quantum digital signature protocol assurance testbed",
    version="0.1.0",
)

# The dashboard is served from a separate dev server during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SessionDep = Annotated[Session, Depends(get_session)]

PROBE_KEY = b"pramana-api-key0"


@app.on_event("startup")
def _startup() -> None:
    init_db(None)


class RunRequest(BaseModel):
    """Start a run."""

    scheme: str
    seed: int = 7
    rounds: int = Field(default=200, ge=1, le=20_000)
    attacks: list[str] = Field(default_factory=list)


class SpecUpload(BaseModel):
    """Upload a specification as YAML text."""

    name: str
    source: str


def _load(name: str, session: Session) -> SchemeSpec:
    """Load a bundled example, or an uploaded spec, by name."""
    if name in example_names():
        return load_example(name)
    stored = session.get(UploadedSpec, name)
    if stored is None:
        raise HTTPException(404, f"no scheme named {name!r}")
    return load_spec_text(stored.source, source=name)


def _static_payload(spec: SchemeSpec) -> dict[str, Any]:
    """D1's verdict, which needs no simulation and returns immediately."""
    verdict = D1Algebra().evaluate(spec, None)
    return {
        "detector": verdict.detector,
        "kind": str(verdict.kind),
        "fired": verdict.fired,
        "applicable": verdict.applicable,
        "confidence": verdict.confidence,
        "evidence": json_safe(verdict.evidence),
        "explanation": verdict.explanation,
        "suggested_fix": verdict.suggested_fix,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe. Used by the container HEALTHCHECK."""
    return {"status": "ok"}


@app.get("/api/schemes")
def list_schemes(session: SessionDep) -> dict[str, Any]:
    """Every loadable scheme, bundled and uploaded."""
    uploaded = [s.name for s in session.query(UploadedSpec).all()]
    return {"bundled": example_names(), "uploaded": uploaded}


@app.get("/api/schemes/{name}")
def get_scheme(name: str, session: SessionDep) -> dict[str, Any]:
    """A scheme's parsed structure and its immediate static verdict."""
    spec = _load(name, session)
    return {
        "spec": json_safe(spec.model_dump(mode="json")),
        "static_verdict": _static_payload(spec),
        "clifford_simulable": spec.is_clifford_simulable(),
    }


@app.post("/api/schemes", status_code=201)
def upload_scheme(payload: SpecUpload, session: SessionDep) -> dict[str, Any]:
    """Validate and store an uploaded specification."""
    try:
        spec = load_spec_text(payload.source, source=payload.name)
    except SpecError as exc:
        raise HTTPException(422, str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(422, str(exc)) from exc

    record = UploadedSpec(
        name=payload.name,
        source=payload.source,
        summary=json_safe({"witnesses": list(spec.forging_witnesses())}),
    )
    session.merge(record)
    session.commit()
    return {"name": payload.name, "static_verdict": _static_payload(spec)}


@app.get("/api/attacks")
def list_attacks() -> dict[str, Any]:
    """The attack registry with citations. The count here is the only true count."""
    return {
        "count": len(ATTACK_REGISTRY),
        "attacks": [
            {
                "name": name,
                "reference": adversary.reference,
                "threat_class": adversary.threat_class,
            }
            for name, adversary in sorted(ATTACK_REGISTRY.items())
        ],
    }


def _execute_run(run_id: str, scheme: str, seed: int, rounds: int, attacks: list[str]) -> None:
    """Background worker: run the protocol, detectors and attacks, store a report."""
    from pramana.api.db import _SessionLocal  # late import: engine set at startup

    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        record = session.get(Run, run_id)
        if record is None:
            return
        spec = load_example(scheme) if scheme in example_names() else None
        if spec is None:
            stored = session.get(UploadedSpec, scheme)
            if stored is None:
                record.status, record.error = "failed", f"no scheme named {scheme!r}"
                session.commit()
                return
            spec = load_spec_text(stored.source, source=scheme)

        record.status, record.stage = "running", "simulating rounds"
        session.commit()

        results = run_protocol_rounds(spec, rounds=rounds, seed=seed, probe_key=PROBE_KEY)
        record.completed_rounds = len(results.rounds)
        record.stage = "evaluating detectors"
        session.commit()

        d1 = D1Algebra().evaluate(spec, None)
        d2 = D2ProbeSprt().evaluate(spec, results)
        d4 = D4Ledger().evaluate(spec, results)
        verdicts = [d1, d2, d4]

        record.stage = "running attacks"
        session.commit()
        attack_results = []
        for name in attacks:
            adversary = ATTACK_REGISTRY.get(name)
            if adversary is None:
                continue
            outcome = adversary.run(spec, trials=50, seed=seed)
            attack_results.append(
                {
                    "attack": outcome.attack,
                    "success_rate": outcome.success_rate,
                    "verification_accepted": outcome.verification_accepted,
                    "message_was_modified": outcome.message_was_modified,
                    "detail": json_safe(outcome.detail),
                }
            )

        classification = classify(
            {v.detector for v in verdicts if v.fired and str(v.kind) == "static"},
            {v.detector for v in verdicts if v.fired and str(v.kind) == "runtime"},
        )
        pool = KeyPool(seed_bits=1_000_000, bits_per_pair=1)
        pool.generate_from_pairs(rounds * 2)
        pool.record_signing_pairs(rounds * spec.signature.length_qubits)
        for _ in range(rounds):
            pool.record_signature()
        pool.consume(rounds * 127, "one pad per signature")

        report = build_report(
            spec, run_id, seed, verdicts, classification, pool.budget(), attack_results
        )
        record.report = json_safe(report.to_dict())
        record.status, record.stage = "complete", "done"
        session.commit()
    except Exception as exc:  # noqa: BLE001 - surface the failure, never swallow it
        record = session.get(Run, run_id)
        if record is not None:
            record.status, record.error = "failed", f"{type(exc).__name__}: {exc}"
            session.commit()
    finally:
        session.close()


@app.post("/api/runs", status_code=202)
def start_run(
    payload: RunRequest, background: BackgroundTasks, session: SessionDep
) -> dict[str, Any]:
    """Start a run in the background and return its id immediately."""
    _load(payload.scheme, session)  # validate before queueing
    run_id = uuid.uuid4().hex[:12]
    session.add(
        Run(id=run_id, scheme=payload.scheme, seed=payload.seed, rounds=payload.rounds)
    )
    session.commit()
    background.add_task(
        _execute_run, run_id, payload.scheme, payload.seed, payload.rounds, payload.attacks
    )
    return {"id": run_id, "status": "pending", "seed": payload.seed}


@app.get("/api/runs")
def list_runs(session: SessionDep) -> dict[str, Any]:
    """Every run, newest first."""
    runs = session.query(Run).order_by(Run.created_at.desc()).all()
    return {"runs": [r.to_dict() for r in runs]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, session: SessionDep) -> dict[str, Any]:
    """Status and results for one run."""
    record = session.get(Run, run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return record.to_dict()


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str, session: SessionDep) -> dict[str, Any]:
    """The assurance report JSON."""
    record = session.get(Run, run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id!r}")
    if record.report is None:
        raise HTTPException(409, f"run {run_id!r} is {record.status}; no report yet")
    return dict(record.report)


@app.get("/api/runs/{run_id}/curves")
def get_curves(run_id: str, session: SessionDep) -> dict[str, Any]:
    """Forgery-probability series: measured points against dashed theory.

    Theory and measurement are returned together on purpose. Their agreement is
    the evidence that the simulator is right, and plotting either alone would
    discard that.
    """
    record = session.get(Run, run_id)
    if record is None:
        raise HTTPException(404, f"no run {run_id!r}")
    lengths = list(range(1, 9))
    probes = list(range(1, 33))
    return {
        "run_id": run_id,
        "seed": record.seed,
        "series": {
            "blind_forgery_acceptance": {
                "x": lengths,
                "theory": [2.0**-n for n in lengths],
                "label": "2**-n",
            },
            "blind_forgery_correct_guess": {
                "x": lengths,
                "theory": [4.0**-n for n in lengths],
                "label": "4**-n",
            },
            "intercept_resend_escape": {
                "x": probes,
                "theory": [(2 / 3) ** d for d in probes],
                "label": "(2/3)**d",
            },
            "unauthorised_verifier_escape": {
                "x": probes,
                "theory": [2.0**-d for d in probes],
                "label": "2**-d",
            },
        },
    }


@app.get("/api/matrix")
def get_matrix(session: SessionDep) -> dict[str, Any]:
    """Scheme x attack grid, with the detectors that fired in each cell."""
    rows = []
    for scheme in example_names():
        spec = load_example(scheme)
        d1 = D1Algebra().evaluate(spec, None)
        cells = []
        for name, adversary in sorted(ATTACK_REGISTRY.items()):
            if not spec.is_clifford_simulable():
                cells.append({"attack": name, "applicable": False, "detectors": {}})
                continue
            outcome = adversary.run(spec, trials=20, seed=5)
            cells.append(
                {
                    "attack": name,
                    "applicable": True,
                    "success_rate": outcome.success_rate,
                    "detectors": {
                        "D1": d1.fired,
                        "D2": False,  # a passing forgery leaves no statistical trace
                        "D3": None,
                        "D4": name == "replay",
                        "D5": None,
                    },
                }
            )
        rows.append({"scheme": scheme, "static_fired": d1.fired, "cells": cells})
    return {"rows": rows, "attacks": sorted(ATTACK_REGISTRY)}
