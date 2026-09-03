"""Runtime verification of every library API this project depends on.

Build plan Part 0, Rule 4: Stim, Qiskit and liboqs change APIs between versions, so
no call in this repository is written from memory. This script imports each library,
records its version, and *exercises the exact calls the later layers need*. It is the
source of truth for API shape -- every layer copies its call patterns from here.

If a call fails, fix it here. Do not work around it downstream.

Run directly for a human-readable report::

    python scripts/verify_env.py

``run_all()`` returns the same results structured, which is what ``tests/test_env.py``
asserts against.

Optional dependency: the post-quantum bridge (Layer 8). See ``check_pqc`` for why it
is opt-in rather than probed by default.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["PASS", "FAIL", "SKIP"]

# Packages that must never be importable in this environment.
# Build plan Appendix B / guarantee G-4: no machine learning anywhere in the
# detection pipeline. This is a constraint from the problem statement, so it is
# enforced mechanically rather than by convention.
BANNED_ML_PACKAGES = (
    "sklearn",
    "torch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "catboost",
    "transformers",
)

# Set to "1" to probe the optional PQC dependency. Off by default -- see check_pqc.
PQC_ENV_FLAG = "PRAMANA_ENABLE_PQC"


@dataclass
class Check:
    """One capability probe."""

    name: str
    required: bool
    status: Status = "SKIP"
    version: str = ""
    detail: str = ""
    notes: list[str] = field(default_factory=list)
    # False for probes that assert a property of the environment rather than
    # exercise a library, and so have no version to report.
    is_library: bool = True

    def ok(self) -> bool:
        """A check passes if it succeeded, or if it is optional and skipped."""
        return self.status == "PASS" or (self.status == "SKIP" and not self.required)


def _fail(check: Check, exc: BaseException) -> Check:
    check.status = "FAIL"
    check.detail = f"{type(exc).__name__}: {exc}"
    return check


def check_python() -> Check:
    """Python 3.12 is the pinned interpreter for this project."""
    c = Check("python", required=True)
    v = sys.version_info
    c.version = f"{v.major}.{v.minor}.{v.micro}"
    c.detail = f"{platform.system()} {platform.machine()}"
    c.status = "PASS" if (v.major, v.minor) >= (3, 12) else "FAIL"
    if c.status == "FAIL":
        c.detail = f"need >=3.12, found {c.version}"
    return c


def check_stim() -> Check:
    """Layer 2a needs: seeded TableauSimulator, h/cnot/x/z, measure, determinism.

    The determinism property is load-bearing (Rule 6), so it is verified here by
    running the same circuit twice under one seed and once under another, rather
    than trusted from the documentation.
    """
    c = Check("stim", required=True)
    try:
        import stim

        c.version = stim.__version__

        def bell_run(seed: int) -> list[tuple[bool, bool]]:
            sim = stim.TableauSimulator(seed=seed)  # seed is keyword-only
            out = []
            for q in range(0, 20, 2):
                sim.h(q)
                sim.cnot(q, q + 1)
                out.append((sim.measure(q), sim.measure(q + 1)))
            return out

        a, b, other = bell_run(42), bell_run(42), bell_run(43)

        # Single-qubit Paulis, needed for the teleportation correction table.
        s = stim.TableauSimulator(seed=1)
        s.x(0)
        s.z(0)
        s.h(0)
        s.measure(0)
        s.measure_many(0)

        problems = []
        if a != b:
            problems.append("same seed produced different results")
        if a == other:
            problems.append("different seeds produced identical results")
        if not all(x == y for x, y in a):
            problems.append("Bell pair measurements were not correlated")

        if problems:
            c.status = "FAIL"
            c.detail = "; ".join(problems)
        else:
            c.status = "PASS"
            c.detail = (
                "TableauSimulator(seed=) h/cnot/x/z/measure/measure_many; "
                "10 Bell pairs correlated; seeding reproducible"
            )
            c.notes.append(
                "Stim seeding is reproducible only for the same Stim version on the "
                "same machine (stim.TableauSimulator docstring). Cross-platform "
                "byte-identity is therefore not claimed; CI pins the version."
            )
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never raises
        return _fail(c, exc)
    return c


def check_qiskit_aer() -> Check:
    """Layer 2b needs a density-matrix backend with depolarizing + amplitude damping.

    Verifies the noise model is actually *attached and effective* -- a noise model
    that silently failed to apply would give a noise engine agreeing with the
    Clifford engine for entirely the wrong reason.
    """
    c = Check("qiskit-aer", required=True)
    try:
        import qiskit
        import qiskit_aer
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import (
            NoiseModel,
            amplitude_damping_error,
            depolarizing_error,
        )

        c.version = f"qiskit-aer {qiskit_aer.__version__} / qiskit {qiskit.__version__}"

        def bell_counts(noise: NoiseModel | None, seed: int) -> dict[str, int]:
            sim = AerSimulator(method="density_matrix", noise_model=noise)
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            job = sim.run(transpile(qc, sim), shots=4000, seed_simulator=seed)
            counts: dict[str, int] = job.result().get_counts()
            return counts

        def broken_correlation(counts: dict[str, int]) -> float:
            """Fraction of shots in which the Bell correlation did not hold."""
            return (counts.get("01", 0) + counts.get("10", 0)) / sum(counts.values())

        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["h", "x", "z"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
        noise.add_all_qubit_quantum_error(amplitude_damping_error(0.02), ["id"])

        clean = bell_counts(None, seed=7)
        dirty_a = bell_counts(noise, seed=7)
        dirty_b = bell_counts(noise, seed=7)

        problems = []
        if broken_correlation(clean) != 0.0:
            problems.append("noiseless run broke the Bell correlation")
        if broken_correlation(dirty_a) <= 0.0:
            problems.append("noise model had no effect on outcomes")
        if dirty_a != dirty_b:
            problems.append("seed_simulator did not reproduce counts")

        if problems:
            c.status = "FAIL"
            c.detail = "; ".join(problems)
        else:
            c.status = "PASS"
            c.detail = (
                f"density_matrix backend; noiseless broken-correlation 0.0, "
                f"noisy {broken_correlation(dirty_a):.4f}; "
                f"reproducible under seed_simulator"
            )
            c.notes.append(
                "Attaching two error channels to one instruction composes them and "
                "emits a warning; that is Aer's documented behaviour, not a fault."
            )
    except Exception as exc:  # noqa: BLE001
        return _fail(c, exc)
    return c


def check_pydantic() -> Check:
    """Layer 1 needs strict models rejecting unknown fields and cross-field violations."""
    c = Check("pydantic", required=True)
    try:
        import pydantic
        from pydantic import (
            BaseModel,
            ConfigDict,
            ValidationError,
            ValidationInfo,
            field_validator,
        )

        c.version = pydantic.VERSION
        if not pydantic.VERSION.startswith("2."):
            c.status = "FAIL"
            c.detail = f"Layer 1 requires pydantic v2, found {pydantic.VERSION}"
            return c

        class Probe(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True)
            count: int
            threshold: int

            @field_validator("threshold")
            @classmethod
            def _threshold_within_count(cls, v: int, info: ValidationInfo) -> int:
                if "count" in info.data and v > info.data["count"]:
                    raise ValueError("threshold exceeds arbitrator count")
                return v

        Probe(count=3, threshold=2)

        rejected: dict[str, str] = {}
        payloads: dict[str, dict[str, Any]] = {
            "threshold>count": {"count": 2, "threshold": 3},
            "unknown field": {"count": 1, "threshold": 1, "surprise": 5},
            "wrong type": {"count": "2", "threshold": 1},
        }
        for label, payload in payloads.items():
            try:
                Probe(**payload)
            except ValidationError as e:
                rejected[label] = e.errors()[0]["type"]

        if len(rejected) != len(payloads):
            missed = sorted(set(payloads) - set(rejected))
            c.status = "FAIL"
            c.detail = f"failed to reject: {', '.join(missed)}"
        else:
            c.status = "PASS"
            c.detail = (
                "extra=forbid, strict types and a cross-field validator all reject "
                f"as required ({', '.join(sorted(rejected.values()))})"
            )
    except Exception as exc:  # noqa: BLE001
        return _fail(c, exc)
    return c


def check_numeric() -> Check:
    """NumPy and SciPy: seeded RNG, and the binomial tail used for test tolerances.

    Rule 3 requires every statistical test to state a tolerance justified by its
    sample size, so scipy.stats.binom is test infrastructure, not an incidental
    dependency.
    """
    c = Check("numpy+scipy", required=True)
    try:
        import numpy as np
        import scipy
        from scipy import stats

        c.version = f"numpy {np.__version__} / scipy {scipy.__version__}"

        first = np.random.default_rng(3).integers(0, 4, 8)
        again = np.random.default_rng(3).integers(0, 4, 8)

        p = 4.0**-4  # blind-forgery success rate at n=4; build plan Layer 5a
        sigma = (p * (1 - p) / 100_000) ** 0.5
        tail = float(stats.binom.sf(500, 100_000, p))

        problems = []
        if not np.array_equal(first, again):
            problems.append("default_rng was not reproducible under a fixed seed")
        if not 0.0 < sigma < 1.0:
            problems.append("binomial sigma out of range")
        if not 0.0 <= tail <= 1.0:
            problems.append("binom.sf returned a non-probability")

        if problems:
            c.status = "FAIL"
            c.detail = "; ".join(problems)
        else:
            c.status = "PASS"
            c.detail = (
                f"default_rng reproducible; binom.sf available; sigma(4^-4, n=1e5) = {sigma:.3e}"
            )
    except Exception as exc:  # noqa: BLE001
        return _fail(c, exc)
    return c


def check_yaml() -> Check:
    """Layer 1 loads scheme specs from YAML, via safe_load only."""
    c = Check("pyyaml", required=True)
    try:
        import yaml

        c.version = yaml.__version__
        source = "name: probe\noperators: [I, X, Y, Z]\nnested:\n  flag: true\n"
        loaded = yaml.safe_load(source)
        expected = {
            "name": "probe",
            "operators": ["I", "X", "Y", "Z"],
            "nested": {"flag": True},
        }
        if loaded != expected:
            c.status = "FAIL"
            c.detail = f"safe_load returned {loaded!r}"
        else:
            c.status = "PASS"
            c.detail = "safe_load round-trips scalars, sequences and nested mappings"
    except Exception as exc:  # noqa: BLE001
        return _fail(c, exc)
    return c


def check_web_stack() -> Check:
    """Layers 9-10: FastAPI under TestClient, and SQLAlchemy over SQLite."""
    c = Check("fastapi+sqlalchemy", required=True)
    try:
        import fastapi
        import sqlalchemy
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine, text

        c.version = f"fastapi {fastapi.__version__} / sqlalchemy {sqlalchemy.__version__}"

        app = FastAPI()

        @app.get("/api/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        response = TestClient(app).get("/api/health")

        engine = create_engine("sqlite://")  # in-memory; Layer 9 uses a file
        with engine.connect() as conn:
            row = conn.execute(text("select 1")).scalar_one()

        if response.status_code != 200 or response.json() != {"status": "ok"} or row != 1:
            c.status = "FAIL"
            c.detail = f"health={response.status_code} {response.text!r}, sqlite={row!r}"
        else:
            c.status = "PASS"
            c.detail = "TestClient serves a route; SQLAlchemy executes against SQLite"
    except Exception as exc:  # noqa: BLE001
        return _fail(c, exc)
    return c


def check_no_ml() -> Check:
    """Guarantee G-4: no machine-learning package may be importable.

    The problem statement prohibits ML in the detection pipeline. A dependency
    arriving transitively would violate that silently, so the environment is
    checked rather than the source.
    """
    c = Check("no-ml-dependencies", required=True, is_library=False)
    present = [m for m in BANNED_ML_PACKAGES if importlib.util.find_spec(m) is not None]
    if present:
        c.status = "FAIL"
        c.detail = (
            f"prohibited package(s) importable: {', '.join(present)}. "
            "Build plan Appendix B forbids any ML dependency."
        )
    else:
        c.status = "PASS"
        c.detail = f"none of {len(BANNED_ML_PACKAGES)} prohibited packages are importable"
    return c


def check_pqc() -> Check:
    """Optional Layer 8 dependency: liboqs, for ML-DSA sign/verify.

    Opt-in via PRAMANA_ENABLE_PQC=1, because ``import oqs`` is not a passive
    import. liboqs-python 0.16.0 runs ``git clone`` of the liboqs C library
    followed by a cmake build *at import time* when no shared library is already
    installed. On a machine without cmake that ends in a RuntimeError after
    several minutes of network and disk activity. Probing it by default would
    make this script slow, network-dependent and side-effecting.

    Per Rule 4, algorithm names are enumerated from the runtime, never hardcoded.
    """
    c = Check("liboqs (PQC bridge)", required=False)
    if os.environ.get(PQC_ENV_FLAG) != "1":
        c.status = "SKIP"
        c.detail = (
            f"not probed; set {PQC_ENV_FLAG}=1 to enable. Layer 8 degrades "
            "gracefully without it (feature flag off, tests skipped)."
        )
        return c
    try:
        import oqs

        mechanisms = list(oqs.get_enabled_sig_mechanisms())
        ml_dsa = [m for m in mechanisms if "ML-DSA" in m]
        c.version = getattr(oqs, "__version__", "unknown")
        c.status = "PASS"
        c.detail = (
            f"{len(mechanisms)} signature mechanisms enumerated; ML-DSA present: {ml_dsa or 'none'}"
        )
        c.notes.append(
            "Mechanism names are enumerated at runtime, never hardcoded (Rule 4). "
            "liboqs is not production-ready per its own maintainers; prototype only."
        )
    except Exception as exc:  # noqa: BLE001
        c.status = "SKIP"
        c.detail = f"unavailable ({type(exc).__name__}: {exc}). Layer 8 feature flag stays off."
    return c


CHECKS: tuple[Callable[[], Check], ...] = (
    check_python,
    check_stim,
    check_qiskit_aer,
    check_pydantic,
    check_numeric,
    check_yaml,
    check_web_stack,
    check_no_ml,
    check_pqc,
)


def run_all() -> list[Check]:
    """Run every capability probe. Never raises; failures come back as results."""
    return [probe() for probe in CHECKS]


def main() -> int:
    """Print the capability report. Returns 0 if every requirement is met."""
    checks = run_all()
    width = max(len(c.name) for c in checks)
    pad = " " * width

    print("Pramana environment verification")
    print("=" * 78)
    for c in checks:
        flag = "" if c.required else "  (optional)"
        print(f"{c.status:4}  {c.name:<{width}}  {c.version or '-'}{flag}")
        print(f"      {pad}  {c.detail}")
        for note in c.notes:
            print(f"      {pad}  note: {note}")
    print("=" * 78)

    failed = [c for c in checks if not c.ok()]
    passed = sum(c.status == "PASS" for c in checks)
    skipped = sum(c.status == "SKIP" for c in checks)
    print(f"{passed} passed, {len(failed)} failed, {skipped} skipped")
    if failed:
        print("\nRequired capabilities missing: " + ", ".join(c.name for c in failed))
        print("Fix these here, in verify_env.py, before writing dependent code (Rule 4).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
