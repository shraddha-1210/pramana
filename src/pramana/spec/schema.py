"""Validated model of a signature scheme specification.

The shape comes from ``docs/derivations.md`` section 5.7, which supersedes the
``encryption`` block in the build plan's Layer 1 schema (ledger V-15). The plan's
block could not distinguish the example schemes from one another, because it
described the operator *set* rather than the encryption *map*.

Validation principles:

* Unknown fields are rejected. A typo must not silently become a default.
* Fields whose only consumer is D1 are still validated here. A specification that
  loads but is meaningless to the detector is a Layer 1 bug, not a Layer 6 one.
* Anything outside the stabilizer class is refused at load with an explanation,
  never accepted and left to fail obscurely inside Layer 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
import stim
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from pramana.spec.kim_forgeability import (
    W_KIM_FORGERY_FREE,
    forging_witnesses_matrix,
    is_forgeable,
    unitary_of,
)
from pramana.spec.operators import (
    BASE_SET,
    REGISTRY,
    NonCliffordOperatorError,
    compose,
    encryption_set,
    forging_witnesses,
    name_of,
    tableau_of,
)

STABILIZER_RANK_LIMITATION = (
    "This project simulates the stabilizer (Clifford) class only, in polynomial time "
    "via the Gottesman-Knill theorem. Going beyond it needs stabilizer-rank "
    "simulation, whose cost grows exponentially in the number of non-Clifford "
    "operations. That technique is designed and documented in the build plan and "
    "deliberately not built -- see docs/honesty_table.md. Building it half-way would "
    "undermine the feasibility argument it exists to support."
)

#: A factor sequence is composed left to right: ["S", "H"] denotes S*H.
FactorSequence = Annotated[list[str], Field(min_length=1)]

# Enum fields are read from YAML as plain strings, so they opt out of strict mode
# individually. Everything else stays strict: an int field still refuses "8", and a
# bool field still refuses 1. Relaxing the whole model would let those through.
Enum = Field(strict=False)


class MessageType(StrEnum):
    """What kind of message the scheme signs."""

    CLASSICAL = "classical"
    STABILIZER = "stabilizer"
    GENERAL = "general"


class OperatorFamily(StrEnum):
    """Declared shape of an operator block, validated against its factors."""

    PAULI = "pauli"
    UV_TYPE = "uv_type"


class Protection(StrEnum):
    """How the classical Bell-measurement outcomes are protected."""

    OTP = "otp"
    WEGMAN_CARTER = "wegman_carter"


class EqualityTest(StrEnum):
    """How verification compares two states."""

    SWAP = "swap"
    PROJECTIVE = "projective"


@dataclass(frozen=True)
class KimVerdict:
    """A forgeable-message verdict, with the theorem that produced it.

    The theorem is part of the verdict, not metadata: Theorem 1 and Theorem 4
    cover different regimes, and a verdict is only auditable if it says which one
    decided it.
    """

    forgeable: bool
    theorem: str
    witness_triple: tuple[int, int, int] | None
    rationale: str


class StrictModel(BaseModel):
    """Base: reject unknown fields, and do not coerce types silently."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OperatorBlock(StrictModel):
    """An operator set of the form ``U {I, X, Y, Z} V``.

    ``family`` is redundant with the factors by construction, and that is the
    point: it is a declared invariant, so a typo in a factor that would silently
    change the scheme's meaning is caught instead.
    """

    family: Annotated[OperatorFamily, Enum]
    left_factor: FactorSequence = Field(default_factory=lambda: ["I"])
    right_factor: FactorSequence = Field(default_factory=lambda: ["I"])

    @field_validator("left_factor", "right_factor")
    @classmethod
    def _known_clifford_names(cls, value: list[str], info: ValidationInfo) -> list[str]:
        for name in value:
            if name not in REGISTRY:
                raise ValueError(
                    f"unknown operator {name!r} in {info.field_name}. Admissible "
                    f"operators are {', '.join(REGISTRY)}. Operator names are not "
                    f"free-form; extend pramana.spec.operators.REGISTRY to add one."
                )
            if not REGISTRY[name].is_clifford:
                raise ValueError(
                    f"{name!r} in {info.field_name} is not a Clifford operator. "
                    f"{REGISTRY[name].note} {STABILIZER_RANK_LIMITATION}"
                )
        return value

    @model_validator(mode="after")
    def _family_matches_factors(self) -> OperatorBlock:
        trivial = all(n == "I" for n in self.left_factor + self.right_factor)
        if self.family is OperatorFamily.PAULI and not trivial:
            raise ValueError(
                "family 'pauli' requires both factors to be identity, but this block "
                f"has left_factor={self.left_factor} right_factor={self.right_factor}. "
                "Declare family 'uv_type' if a non-identity factor is intended."
            )
        if self.family is OperatorFamily.UV_TYPE and trivial:
            raise ValueError(
                "family 'uv_type' requires at least one non-identity factor, but both "
                "factors are identity. Declare family 'pauli' if that is intended."
            )
        return self

    def operator_set(self) -> tuple[stim.Tableau, ...]:
        """The four operators ``U P V`` this block denotes."""
        return encryption_set(self.left_factor, self.right_factor)

    def is_identity_sandwich(self) -> bool:
        """Whether both factors compose to the identity."""
        identity = stim.Tableau(1)
        return bool(
            compose(self.left_factor) == identity and compose(self.right_factor) == identity
        )


class Entanglement(StrictModel):
    """The entanglement resource consumed per signature qubit."""

    resource: Literal["bell", "ghz3"]
    pairs_per_signature_qubit: int = Field(ge=1)


class Signature(StrictModel):
    """Signature geometry."""

    length_qubits: int = Field(ge=1)


class ClassicalOutcomes(StrictModel):
    """The classical Bell-measurement string ``M_A`` and how it is protected.

    ``used_in_verification`` selects which Choi attack variant applies: variant 1
    when Trent does not use ``M_A`` in his validity test, variant 2 when he does
    (``docs/derivations.md`` section 5.2). It is not a fix dimension.

    ``protection: otp`` is a *vulnerability*, not a neutral choice: a one-time pad
    gives confidentiality and no integrity, so ``M_A`` is bit-flip malleable
    (ledger V-07).
    """

    used_in_verification: bool
    protection: Annotated[Protection, Enum]


class Binding(StrictModel):
    """Whether qubit position is bound into the signature (D1c).

    Without positional binding an attacker can permute message and signature
    qubits -- Choi section III C. Structured financial and government documents
    have predictable layouts, so reordering dates and amounts is a real attack.
    """

    positional: bool


class Arbitrator(StrictModel):
    """Arbitration parameters. Detection at ``count >= 2``, identification at 3."""

    count: int = Field(ge=1)
    threshold: int = Field(ge=1)
    trust: Literal["semi_trusted", "untrusted"]

    @model_validator(mode="after")
    def _threshold_within_count(self) -> Arbitrator:
        if self.threshold > self.count:
            raise ValueError(
                f"threshold ({self.threshold}) exceeds arbitrator count ({self.count}). "
                "Set threshold <= count."
            )
        return self


class Probes(StrictModel):
    """Probe-round configuration."""

    enabled: bool
    basis_set: Literal["six_state", "four_state"]
    rounds_per_signature: int = Field(ge=0)


class Verification(StrictModel):
    """How a signature is corrected and compared.

    ``correction_rule`` survives from the build plan's schema; Layer 2 needs it.
    The named convention is *not yet established* -- the Bell-outcome to Pauli
    correction mapping differs between references and is being determined
    empirically at Layer 2 (ledger V-16), not written from memory.
    """

    correction_rule: Literal["pauli_standard"]
    equality_test: Annotated[EqualityTest, Enum]


class SchemeSpec(StrictModel):
    """A complete, validated signature scheme specification."""

    name: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    message_type: Annotated[MessageType, Enum]
    entanglement: Entanglement
    signature: Signature
    rotation: OperatorBlock
    rotation_count: int = Field(ge=1)
    signing_encryption: OperatorBlock
    assistant_unitary: str | None = None
    classical_outcomes: ClassicalOutcomes
    binding: Binding
    arbitrator: Arbitrator
    probes: Probes
    verification: Verification

    @field_validator("message_type")
    @classmethod
    def _reject_general_messages(cls, value: MessageType) -> MessageType:
        if value is MessageType.GENERAL:
            raise NotImplementedError(
                f"message_type 'general' is not supported. {STABILIZER_RANK_LIMITATION}"
            )
        return value

    @model_validator(mode="after")
    def _projective_test_requires_classical_message(self) -> SchemeSpec:
        """Ledger V-21 -- derived by us, not from Choi.

        Choi section III C says the swap test is the only known method for comparing
        two general quantum states. A projective comparison is sound for a classical
        message and not for an arbitrary quantum one, so the combination is refused
        rather than left to produce a meaningless verdict at Layer 3.
        """
        if (
            self.verification.equality_test is EqualityTest.PROJECTIVE
            and self.message_type is not MessageType.CLASSICAL
        ):
            raise ValueError(
                f"equality_test 'projective' requires message_type 'classical', but "
                f"message_type is {self.message_type.value!r}. Choi section III C notes "
                "the swap test is the only known equality test for general quantum "
                "states; a projective comparison is sound only for classical messages. "
                "This constraint is derived by this project, not taken from the paper "
                "(ledger V-21)."
            )
        return self

    @field_validator("assistant_unitary")
    @classmethod
    def _assistant_is_a_registry_name(cls, value: str | None) -> str | None:
        """The assistant unitary is a registry name, never a free-form string.

        Unlike a factor it *may* be non-Clifford: D1 is static and never
        simulates, so a non-Clifford assistant is auditable even though it cannot
        run on the Clifford engine.
        """
        if value is not None and value not in REGISTRY:
            raise ValueError(
                f"unknown assistant_unitary {value!r}. Admissible operators are "
                f"{', '.join(REGISTRY)}. Operator names are not free-form; extend "
                f"pramana.spec.operators.REGISTRY to add one."
            )
        return value

    @model_validator(mode="after")
    def _assistant_and_factors_are_alternatives(self) -> SchemeSpec:
        """A scheme declares its assistant one way or the other, never both.

        ``signing_encryption`` factors and ``assistant_unitary`` are two ways of
        naming the same operator. Allowing both would create two sources of truth
        for what D1 audits -- the failure the ``family`` invariant exists to
        prevent (Q-9).
        """
        if (
            self.assistant_unitary is not None
            and not self.signing_encryption.is_identity_sandwich()
        ):
            raise ValueError(
                f"assistant_unitary {self.assistant_unitary!r} is declared, but "
                f"signing_encryption also carries non-identity factors "
                f"(left={self.signing_encryption.left_factor}, "
                f"right={self.signing_encryption.right_factor}). These are two ways "
                "of naming the same operator; declare exactly one of them."
            )
        return self

    def is_clifford_simulable(self) -> bool:
        """Whether the runtime engines can simulate this scheme.

        False when the scheme declares a non-Clifford assistant unitary. Such a
        scheme is still fully **auditable**: D1 is static, reads four real
        coefficients and never simulates. The restriction constrains the runtime
        layers only, and it is a scope fact (ledger V-27), not a defect.
        """
        return self.assistant_unitary is None or REGISTRY[self.assistant_unitary].is_clifford

    def assistant_matrix(self) -> np.ndarray:
        """The single assistant unitary this scheme adjoins to the Pauli set.

        Taken from the declared ``assistant_unitary`` when present, otherwise
        composed from the signing-encryption factors.

        Raises:
            NotImplementedError: Both factors are non-identity. Kim's Theorem 4 is
                stated for a single assistant unitary; which composite plays that
                role in a two-sided sandwich is not derived in any source we have
                read, and is not guessed here (Rule 1).
        """
        if self.assistant_unitary is not None:
            if self.assistant_unitary == "W_kim_forgery_free":
                return W_KIM_FORGERY_FREE
            return unitary_of(tableau_of(self.assistant_unitary))

        left = compose(self.signing_encryption.left_factor)
        right = compose(self.signing_encryption.right_factor)
        identity = stim.Tableau(1)
        if left != identity and right != identity:
            raise NotImplementedError(
                "Kim Theorem 4 is stated for a single assistant unitary, but this "
                "scheme has non-identity factors on both sides. Required source: a "
                "derivation of the two-sided case. Not guessed here."
            )
        return unitary_of(left * right)

    def encryption_matrices(self) -> tuple[np.ndarray, ...]:
        """The encryption operator set as matrices: ``P * assistant`` for each Pauli."""
        assistant = self.assistant_matrix()
        return tuple(unitary_of(tableau_of(p)) @ assistant for p in BASE_SET)

    def rotation_matrices(self) -> tuple[np.ndarray, ...]:
        """The rotation operator set as matrices."""
        return tuple(unitary_of(t) for t in self.rotation.operator_set())

    def forging_witnesses(self) -> tuple[str, ...]:
        """Non-trivial operators that make Choi-class existential forgery succeed.

        The D1 predicate of ``docs/derivations.md`` section 5.3. It decides the
        **universal-Pauli-commutant class** only. Empty means not vulnerable *to
        that class*, which is strictly weaker than secure -- see ``kim_verdict``,
        which is necessary and sufficient and catches schemes this search clears.

        Two code paths, and which one ran matters for how the result is reported:

        * Clifford scheme -> exact tableau search, no tolerance.
        * Declared non-Clifford assistant -> matrix search against a tolerance,
          because a non-Clifford operator has no tableau. Numerically decided, not
          exactly decided. Ledger V-37.

        Returns:
            Registry names of the witnesses, sorted.
        """
        if self.is_clifford_simulable():
            witnesses = forging_witnesses(
                self.signing_encryption.operator_set(), self.rotation.operator_set()
            )
            return tuple(sorted(name_of(q) for q in witnesses))
        return forging_witnesses_matrix(self.encryption_matrices(), self.rotation_matrices())

    def kim_verdict(self) -> KimVerdict:
        """Whether a forgeable quantum message exists. Kim, Lee & Lee (2018).

        **Branches on the number of random rotations**, because the two theorems
        cover different regimes and applying the wrong one gives the right answer
        for the wrong reason:

        * ``rotation_count == 2`` -> **Theorem 1**. A forgeable message exists for
          *any* assistant unitary, unconditionally. The alpha/beta/gamma condition
          is not evaluated, because it does not decide this case.
        * ``rotation_count >= 3`` -> **Theorem 4**. The alpha/beta/gamma condition
          is necessary and sufficient.

        Raises:
            NotImplementedError: ``rotation_count`` is 1. Neither theorem we have
                read covers a single rotation, and the answer is not guessed.
        """
        if self.rotation_count < 2:
            raise NotImplementedError(
                f"rotation_count is {self.rotation_count}. Kim Theorem 1 covers two "
                "random rotations and Theorem 4 covers three or more; neither covers "
                "one. Required source: a treatment of the single-rotation case. Not "
                "guessed here."
            )
        if self.rotation_count == 2:
            return KimVerdict(
                forgeable=True,
                theorem="Theorem 1",
                witness_triple=None,
                rationale=(
                    "Two random rotations: a forgeable quantum message exists for any "
                    "assistant unitary, unconditionally. The assistant unitary is not "
                    "examined, because it cannot change the answer."
                ),
            )

        forgeable, triple = is_forgeable(self.assistant_matrix())
        if forgeable:
            rationale = (
                f"Three or more random rotations, so Theorem 4 decides. The product "
                f"alpha_l*beta_m*gamma_n vanishes for (l, m, n) = {triple}, placing the "
                f"assistant unitary in W_{triple[0]}{triple[1]}{triple[2]}."
                if triple
                else "Theorem 4 condition satisfied."
            )
        else:
            rationale = (
                "Three or more random rotations, so Theorem 4 decides. No product "
                "alpha_l*beta_m*gamma_n vanishes, so no forgeable quantum message "
                "exists for this assistant unitary."
            )
        return KimVerdict(
            forgeable=forgeable,
            theorem="Theorem 4",
            witness_triple=triple,
            rationale=rationale,
        )


__all__ = [
    "STABILIZER_RANK_LIMITATION",
    "Arbitrator",
    "Binding",
    "ClassicalOutcomes",
    "Entanglement",
    "EqualityTest",
    "MessageType",
    "NonCliffordOperatorError",
    "OperatorBlock",
    "OperatorFamily",
    "Probes",
    "Protection",
    "SchemeSpec",
    "Signature",
    "Verification",
]
