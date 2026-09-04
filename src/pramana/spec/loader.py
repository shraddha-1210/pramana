"""Load scheme specifications from YAML into validated models.

``yaml.safe_load`` only -- a specification is untrusted input, and the full loader
can construct arbitrary Python objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pramana.spec.schema import SchemeSpec

EXAMPLES_DIR = Path(__file__).parent / "examples"


class SpecError(ValueError):
    """A specification could not be loaded.

    Carries the underlying message rather than a generic failure: an error must say
    what happened and what to do about it.
    """


def load_spec_text(text: str, *, source: str = "<string>") -> SchemeSpec:
    """Parse and validate a specification from YAML text.

    Raises:
        SpecError: The YAML is malformed, is not a mapping, or fails validation.
        NotImplementedError: The specification requests a documented, deliberate
            limitation -- currently ``message_type: general``. Propagated unwrapped,
            because it means "this is out of scope", not "this input is malformed".
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{source}: not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise SpecError(
            f"{source}: expected a mapping at the top level, found {type(raw).__name__}."
        )

    try:
        return SchemeSpec(**raw)
    except NotImplementedError:
        raise
    except ValidationError as exc:
        raise SpecError(f"{source}: {_render(exc)}") from exc
    except TypeError as exc:
        raise SpecError(f"{source}: {exc}") from exc


def load_spec(path: str | Path) -> SchemeSpec:
    """Load and validate a specification from a YAML file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read specification {p}: {exc}") from exc
    return load_spec_text(text, source=str(p))


def load_example(name: str) -> SchemeSpec:
    """Load one of the bundled example specifications by bare name."""
    path = EXAMPLES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise SpecError(
            f"no example specification named {name!r}. Available: {', '.join(example_names())}."
        )
    return load_spec(path)


def example_names() -> list[str]:
    """Names of the bundled example specifications, sorted."""
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.yaml"))


def _render(exc: ValidationError) -> str:
    """Render a pydantic error as one actionable line per problem."""
    parts: list[str] = []
    for err in exc.errors():
        location = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def spec_summary(spec: SchemeSpec) -> dict[str, Any]:
    """A flat, display-ready summary. Used by the Scheme Lab view at Layer 10."""
    witnesses = spec.forging_witnesses()
    return {
        "name": spec.name,
        "reference": spec.reference,
        "message_type": spec.message_type.value,
        "signing_encryption": (
            f"{spec.signing_encryption.family.value} "
            f"U={spec.signing_encryption.left_factor} "
            f"V={spec.signing_encryption.right_factor}"
        ),
        "witness_count": len(witnesses),
    }
