"""Layer 0 exit gate: every required library capability is present and working.

These tests assert against ``scripts/verify_env.py``, which is the project's single
source of truth for library API shape (build plan Rule 4). They deliberately do not
re-probe the libraries themselves -- duplicating the probes here would create a second
place for API drift to hide.

Optional capabilities skip; they never fail.
"""

from __future__ import annotations

import pytest
from verify_env import BANNED_ML_PACKAGES, Check, run_all

# One probe run shared by the whole module. Each probe exercises real simulation
# work, so re-running per test would make the gate needlessly slow.
RESULTS: list[Check] = run_all()
BY_NAME: dict[str, Check] = {c.name: c for c in RESULTS}

REQUIRED = [c.name for c in RESULTS if c.required]
OPTIONAL = [c.name for c in RESULTS if not c.required]


def test_every_probe_reports_a_valid_status() -> None:
    """A probe must resolve to one of the three known states, never crash."""
    assert RESULTS, "verify_env exposed no checks"
    for check in RESULTS:
        assert check.status in {"PASS", "FAIL", "SKIP"}, f"{check.name}: {check.status}"
        assert check.detail, f"{check.name} reported no detail"


@pytest.mark.parametrize("name", REQUIRED)
def test_required_capability_present(name: str) -> None:
    """Every required capability passes. This is the Layer 0 exit gate."""
    check = BY_NAME[name]
    assert check.status == "PASS", f"{name} did not pass: {check.detail}"
    if check.is_library:
        # Versions are recorded in docs/verification.md; a library probe that
        # passes without reporting one leaves the ledger unfalsifiable.
        assert check.version, f"{name} passed without recording a version"


@pytest.mark.parametrize("name", OPTIONAL)
def test_optional_capability_absent_is_not_a_failure(name: str) -> None:
    """An optional capability may be missing; it may not be broken."""
    check = BY_NAME[name]
    if check.status == "SKIP":
        pytest.skip(f"{name} unavailable: {check.detail}")
    assert check.status == "PASS", f"{name} is present but failing: {check.detail}"


def test_no_machine_learning_dependency_is_importable() -> None:
    """Guarantee G-4 and Appendix B: no ML anywhere, enforced against the environment.

    Called out as its own test rather than left inside the parametrised sweep
    because it is a constraint from the problem statement. If it ever fails, the
    failure must name itself unambiguously in the CI log.
    """
    check = BY_NAME["no-ml-dependencies"]
    assert check.status == "PASS", check.detail
    assert len(BANNED_ML_PACKAGES) >= 8


def test_verify_env_exit_code_is_zero_when_requirements_met() -> None:
    """`python scripts/verify_env.py` must exit 0, since CI gates on it."""
    failed = [c.name for c in RESULTS if not c.ok()]
    assert not failed, f"required capabilities missing: {', '.join(failed)}"
