"""Layer 1 exit gate: the schema, the operator registry, and the three examples.

Every assertion here is an analytically derived value, never a captured output
(build plan Rule 3). The witness counts come from the algebra in
``docs/derivations.md`` sections 5.3 and 5.3a, not from running the code and
recording what it printed.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from pramana.spec.kim_forgeability import (
    W_KIM_FORGERY_FREE,
    ZERO_TOLERANCE,
    is_forgeable,
    smallest_nonzero_term,
    su2_components,
    theorem4_terms,
    unitary_of,
)
from pramana.spec.loader import SpecError, example_names, load_example, load_spec_text
from pramana.spec.operators import (
    BASE_SET,
    CLIFFORD_NAMES,
    REGISTRY,
    NonCliffordOperatorError,
    clifford_group,
    commutes_up_to_phase,
    compose,
    encryption_set,
    forging_witnesses,
    name_of,
    tableau_of,
)
from pramana.spec.schema import (
    EqualityTest,
    MessageType,
    OperatorFamily,
    SchemeSpec,
)

# --------------------------------------------------------------------------
# The group. Generated, never assumed.
# --------------------------------------------------------------------------


def test_clifford_group_has_order_24() -> None:
    """|C_1/U(1)| = 24.

    The single-qubit Clifford group modulo global phase is isomorphic to S_4.
    The order is *generated* here by closure from {H, S}. If this assertion
    fails, the representation is wrong and everything D1 computes on top of it
    is meaningless -- fix the representation, never the number.
    """
    assert len(clifford_group()) == 24


def test_clifford_group_is_closed_and_matches_stim_enumeration() -> None:
    """The generated set is a group, and it is the whole group."""
    group = clifford_group()
    keys = {str(g) for g in group}
    assert len(keys) == 24, "closure produced duplicate elements"
    assert keys == {str(t) for t in stim.Tableau.iter_all(1)}
    assert all(str(a * b) in keys for a in group for b in group), "not closed under product"
    assert all(str(a**-1) in keys for a in group), "not closed under inverse"


def test_every_clifford_registry_entry_is_in_the_group() -> None:
    """The admissible vocabulary all lies inside C_1/U(1)."""
    keys = {str(g) for g in clifford_group()}
    for name in CLIFFORD_NAMES:
        assert str(tableau_of(name)) in keys, f"{name} is not in the generated group"


# --------------------------------------------------------------------------
# The algebra the whole finding rests on.
# --------------------------------------------------------------------------


def test_hadamard_conjugation_negates_pauli_y() -> None:
    """H Y H = -Y, verified against explicit matrices.

    This single identity is why Choi's published fix does not clear his own
    predicate: Y anticommutes with H, and anticommutation is commutation up to
    the global phase -1 -- the exact notion the attack in
    ``docs/derivations.md`` section 5.2 relies on. Ledger V-18.

    Asserted against exact matrix algebra rather than the tableau
    representation, so that a fault in the representation cannot make this pass.
    """
    h = np.sqrt(0.5) * np.array([[1, 1], [1, -1]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    assert np.allclose(h @ y @ h, -y), "H Y H != -Y"
    assert np.allclose(h @ y, -(y @ h)), "H and Y do not anticommute"
    # Anticommuting is commuting up to global phase, so the quotient sees them
    # as commuting -- which is precisely what lets Y survive the fix.
    assert commutes_up_to_phase(tableau_of("H"), tableau_of("Y"))


def test_hadamard_does_not_commute_with_x_or_z_even_up_to_phase() -> None:
    """H genuinely breaks commutation with X and Z. Only Y slips through."""
    assert not commutes_up_to_phase(tableau_of("H"), tableau_of("X"))
    assert not commutes_up_to_phase(tableau_of("H"), tableau_of("Z"))


def test_all_paulis_pairwise_commute_up_to_phase() -> None:
    """The basis of Choi's attack: every Pauli pair commutes up to a sign."""
    for a in BASE_SET:
        for b in BASE_SET:
            assert commutes_up_to_phase(tableau_of(a), tableau_of(b))


def test_factor_sequence_composition_order() -> None:
    """``["S", "H"]`` means S*H, and ``stim`` A*B corresponds to matrix A@B.

    Composition order is determined at runtime, not taken from documentation
    (Rule 4, ledger V-20). The two orders are distinguishable: S*H and H*S
    induce different permutations of the Pauli axes.
    """
    assert compose(["S", "H"]) == tableau_of("S") * tableau_of("H")
    assert compose(["S", "H"]) != compose(["H", "S"])
    assert compose([]) == stim.Tableau(1)
    assert compose(["I", "I"]) == stim.Tableau(1)

    def axis_action(t: stim.Tableau) -> dict[str, str]:
        return {p: str(t(stim.PauliString(p))).lstrip("+-_") for p in ("X", "Y", "Z")}

    # S*H maps X->Z, Y->X, Z->Y; H*S maps X->Y, Y->Z, Z->X. Both are 3-cycles,
    # and they are inverses of one another, so order is observable.
    assert axis_action(compose(["S", "H"])) == {"X": "Z", "Y": "X", "Z": "Y"}
    assert axis_action(compose(["H", "S"])) == {"X": "Y", "Y": "Z", "Z": "X"}


# --------------------------------------------------------------------------
# The D1 predicate on the three schemes.
# --------------------------------------------------------------------------


def test_baseline_has_three_witnesses_being_the_non_identity_paulis() -> None:
    """(I,I)-type: every non-identity Pauli forges. Derivation: §5.3.

    When E and R are both Pauli, any Pauli Q commutes with all of them up to a
    sign, so the witnesses are exactly X, Y and Z -- three of them, no more,
    since a non-Pauli Clifford cannot commute with the whole Pauli group.
    """
    spec = load_example("baseline_bell_aqs")
    witnesses = spec.forging_witnesses()
    assert len(witnesses) == 3
    assert sorted(name_of(q) for q in witnesses) == ["X", "Y", "Z"]


def test_choi_fixed_still_has_exactly_one_witness_and_it_is_y() -> None:
    """(I,H)-type: Choi's published fix leaves Q = Y. Derivation: §5.3a, V-18.

    Not a bug and not to be repaired by changing the factor: this scheme is
    defined as "the paper's fix, applied". H fixes the Y axis under conjugation
    (H Y H = -Y), and a Pauli fixed up to sign is precisely a witness. X and Z
    are swapped by H and so are eliminated, leaving exactly one.

    The gap itself is published -- Zhang et al. (2013) are credited with showing
    (I,H)-type encryption still insecure. This is reproduction, not discovery.
    """
    spec = load_example("choi_fixed_aqs")
    witnesses = spec.forging_witnesses()
    assert len(witnesses) == 1
    assert name_of(witnesses[0]) == "Y"
    assert witnesses[0] == tableau_of("Y")


def test_pauli_witness_free_scheme_has_no_pauli_witnesses() -> None:
    """(I, S*H)-type: no witness survives. Derivation: §5.3a, V-19.

    S*H permutes the Pauli axes as a 3-cycle (X->Z->Y->X), fixing none. A
    witness must be a Pauli fixed up to sign by conjugation with V, so a
    fixed-point-free V admits none.

    Zero Pauli witnesses does NOT mean secure. The same scheme is forgeable
    under Kim Theorem 4 -- see test_pauli_witness_free_scheme_is_still_forgeable.
    """
    spec = load_example("pauli_witness_free_aqs")
    assert spec.forging_witnesses() == ()


def test_exactly_eight_of_the_group_are_pauli_witness_free_right_factors() -> None:
    """The fixed-point-free count, derived rather than observed. Ledger V-19.

    A witness-free V must fix no Pauli axis under conjugation. Conjugation by a
    Clifford permutes the three axes, and the fixed-point-free permutations of
    three objects are the two 3-cycles. In S_4 the elements of order 3 number 8,
    so exactly 8 of the 24 qualify.
    """
    rotation = encryption_set(["I"], ["I"])
    clean = [
        v
        for v in clifford_group()
        if not forging_witnesses(tuple(p * v for p in rotation), rotation)
    ]
    assert len(clean) == 8


def test_witness_counts_across_all_three_examples() -> None:
    """The comparison table in §5.7, asserted end to end: 3, 1, 0."""
    counts = {name: len(load_example(name).forging_witnesses()) for name in example_names()}
    assert counts == {
        "baseline_bell_aqs": 3,
        "choi_fixed_aqs": 1,
        "pauli_witness_free_aqs": 0,
    }


# --------------------------------------------------------------------------
# Loading and round-tripping.
# --------------------------------------------------------------------------


def test_all_three_examples_are_present_and_load() -> None:
    """The exit gate: every bundled example parses and validates."""
    assert example_names() == ["baseline_bell_aqs", "choi_fixed_aqs", "pauli_witness_free_aqs"]
    for name in example_names():
        spec = load_example(name)
        assert isinstance(spec, SchemeSpec)
        assert spec.name == name


@pytest.mark.parametrize("name", ["baseline_bell_aqs", "choi_fixed_aqs", "pauli_witness_free_aqs"])
def test_every_field_round_trips(name: str) -> None:
    """Dumping and reloading a spec yields an identical model."""
    spec = load_example(name)
    assert SchemeSpec(**spec.model_dump()) == spec


def test_pauli_witness_free_declares_its_provenance() -> None:
    """`witness_free_aqs` must never read as the paper's fix. Ledger V-22."""
    spec = load_example("pauli_witness_free_aqs")
    assert "Derived by this project" in spec.reference
    assert "NOT from Choi" in spec.reference


def test_choi_fixed_separates_the_paper_s_fix_from_our_additions() -> None:
    """Only `signing_encryption` is Choi's; the reference field says so."""
    reference = load_example("choi_fixed_aqs").reference
    assert "Only the signing_encryption block is Choi's" in reference


# --------------------------------------------------------------------------
# Rejection. Each case names what to do about it.
# --------------------------------------------------------------------------


def _spec_dict(**overrides: object) -> dict[str, object]:
    """A minimal valid spec as a plain dict, with overrides applied."""
    base = load_example("baseline_bell_aqs").model_dump(mode="json")
    base.update(overrides)
    return base


def test_general_message_type_raises_not_implemented() -> None:
    """`message_type: general` is a documented limitation, not a malformed input."""
    with pytest.raises(NotImplementedError, match="stabilizer-rank"):
        SchemeSpec(**_spec_dict(message_type="general"))


def test_non_clifford_factor_is_rejected_with_the_limitation_named() -> None:
    """A T factor is refused at load, not left to fail inside Layer 2."""
    spec = _spec_dict()
    spec["signing_encryption"] = {
        "family": "uv_type",
        "left_factor": ["I"],
        "right_factor": ["T"],
    }
    with pytest.raises(SpecError, match="not a Clifford operator"):
        load_spec_text(_as_yaml(spec))


def test_t_is_registered_but_not_clifford_and_has_no_tableau() -> None:
    """T is parsed and then rejected, rather than being an unknown name."""
    assert "T" in REGISTRY
    assert REGISTRY["T"].is_clifford is False
    assert REGISTRY["T"].tableau is None
    with pytest.raises(NonCliffordOperatorError, match="non-Clifford"):
        tableau_of("T")


def test_unknown_operator_name_is_rejected() -> None:
    """Operator names are not free-form."""
    with pytest.raises(KeyError, match="Unknown operator"):
        tableau_of("W")


@pytest.mark.parametrize(
    ("family", "left", "right"),
    [
        ("pauli", ["I"], ["H"]),  # declared Pauli, but a factor is not identity
        ("uv_type", ["I"], ["I"]),  # declared (U,V)-type, but both factors trivial
    ],
)
def test_family_must_agree_with_its_factors(family: str, left: list[str], right: list[str]) -> None:
    """The declared invariant catches a factor typo. Q-9."""
    spec = _spec_dict()
    spec["signing_encryption"] = {
        "family": family,
        "left_factor": left,
        "right_factor": right,
    }
    with pytest.raises(SpecError, match="family"):
        load_spec_text(_as_yaml(spec))


def test_threshold_above_count_is_rejected() -> None:
    """Arbitration threshold cannot exceed the number of arbitrators."""
    spec = _spec_dict()
    spec["arbitrator"] = {"count": 2, "threshold": 3, "trust": "semi_trusted"}
    with pytest.raises(SpecError, match="threshold"):
        load_spec_text(_as_yaml(spec))


@pytest.mark.parametrize(
    ("block", "payload"),
    [
        ("signature", {"length_qubits": 0}),
        ("signature", {"length_qubits": -8}),
        ("entanglement", {"resource": "bell", "pairs_per_signature_qubit": 0}),
    ],
)
def test_non_positive_sizes_are_rejected(block: str, payload: dict[str, object]) -> None:
    """Lengths and counts must be positive."""
    spec = _spec_dict()
    spec[block] = payload
    with pytest.raises(SpecError):
        load_spec_text(_as_yaml(spec))


def test_invalid_entanglement_resource_is_rejected() -> None:
    """An entanglement resource outside the supported set is refused."""
    spec = _spec_dict()
    spec["entanglement"] = {"resource": "w_state", "pairs_per_signature_qubit": 1}
    with pytest.raises(SpecError, match="resource"):
        load_spec_text(_as_yaml(spec))


def test_unknown_field_is_rejected() -> None:
    """A typo must not silently become a default."""
    with pytest.raises(SpecError, match="surprise"):
        load_spec_text(_as_yaml(_spec_dict(surprise=1)))


def test_projective_equality_test_requires_a_classical_message() -> None:
    """Ledger V-21, derived by us: swap is the only general quantum equality test."""
    spec = _spec_dict(message_type="stabilizer")
    spec["verification"] = {"correction_rule": "pauli_standard", "equality_test": "projective"}
    with pytest.raises(SpecError, match="projective"):
        load_spec_text(_as_yaml(spec))


def test_swap_equality_test_is_allowed_for_a_stabilizer_message() -> None:
    """The V-21 constraint restricts only the projective test."""
    spec = _spec_dict(message_type="stabilizer")
    spec["verification"] = {"correction_rule": "pauli_standard", "equality_test": "swap"}
    loaded = load_spec_text(_as_yaml(spec))
    assert loaded.message_type is MessageType.STABILIZER
    assert loaded.verification.equality_test is EqualityTest.SWAP


def test_malformed_yaml_reports_the_source() -> None:
    """A YAML syntax error names the source document."""
    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec_text("name: [unclosed\n", source="broken.yaml")


def test_non_mapping_document_is_rejected() -> None:
    """A spec must be a mapping, not a list or scalar."""
    with pytest.raises(SpecError, match="expected a mapping"):
        load_spec_text("- just\n- a\n- list\n")


def test_missing_example_names_the_available_ones() -> None:
    """An unknown example name lists what is available."""
    with pytest.raises(SpecError, match="baseline_bell_aqs"):
        load_example("no_such_scheme")


def test_error_messages_say_what_to_do() -> None:
    """Errors state what happened and how to fix it, never just that it failed."""
    spec = _spec_dict()
    spec["arbitrator"] = {"count": 2, "threshold": 3, "trust": "semi_trusted"}
    with pytest.raises(SpecError) as excinfo:
        load_spec_text(_as_yaml(spec))
    assert "Set threshold <= count" in str(excinfo.value)


def test_strictness_survives_the_enum_relaxation() -> None:
    """Enum fields accept strings; int and bool fields still do not coerce."""
    with pytest.raises(SpecError):
        load_spec_text(_as_yaml(_spec_dict(signature={"length_qubits": "8"})))
    with pytest.raises(SpecError):
        load_spec_text(_as_yaml(_spec_dict(binding={"positional": 1})))


def test_operator_family_values_are_the_documented_two() -> None:
    """The family vocabulary is exactly the two values in section 5.7."""
    assert {f.value for f in OperatorFamily} == {"pauli", "uv_type"}


def _as_yaml(data: dict[str, object]) -> str:
    """Serialise a spec dict back to YAML so the loader path is exercised."""
    import yaml

    return yaml.safe_dump(data, sort_keys=False)


# --------------------------------------------------------------------------
# Kim, Lee & Lee Theorem 4. A separate, strictly stronger criterion.
# Source: arXiv:1708.05111.
# --------------------------------------------------------------------------


def test_kim_reproduces_their_own_forgery_free_operator() -> None:
    """Kim's forgery-free W is not forgeable. Validates our implementation.

    This is the strongest available check on the Theorem 4 code: the paper names
    an operator for which no forgeable message exists, and our independent
    implementation must agree. w = (0, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)) gives
    alpha_l = -1/6, beta_m = 1/3, gamma_n = -1/3, all non-zero, for every index.
    """
    w = su2_components(W_KIM_FORGERY_FREE)
    assert w == pytest.approx((0.0, 3**-0.5, 3**-0.5, 3**-0.5), abs=1e-9)

    alpha, beta, gamma = theorem4_terms(w)
    assert all(v == pytest.approx(-1 / 6, abs=1e-9) for v in alpha.values())
    assert all(v == pytest.approx(1 / 3, abs=1e-9) for v in beta.values())
    assert all(v == pytest.approx(-1 / 3, abs=1e-9) for v in gamma.values())

    forgeable, triple = is_forgeable(W_KIM_FORGERY_FREE)
    assert forgeable is False
    assert triple is None


def test_hadamard_is_in_w123_because_alpha_is_zero() -> None:
    """Kim section III: H is in W_123 because alpha_1 = 0. Their [10] is Choi 2011.

    H normalised to SU(2) has w0 = 0 and two components of magnitude 1/sqrt(2),
    so alpha for those indices is 0 + 1/2 - 1/2 = 0. This is the published route
    to the same conclusion our Pauli-witness search reaches via Q = Y.
    """
    w = su2_components(unitary_of(tableau_of("H")))
    alpha, _, _ = theorem4_terms(w)
    assert min(abs(v) for v in alpha.values()) == pytest.approx(0.0, abs=1e-6)
    assert is_forgeable(unitary_of(tableau_of("H")))[0] is True


def test_s_times_h_has_all_three_alphas_zero() -> None:
    """S*H normalises to w = (1/2, 1/2, -1/2, 1/2), so every alpha_l vanishes.

    alpha_l = 1/4 + 1/4 - 1/2 = 0 for l in {1,2,3}: S*H is in W_123 and is
    forgeable, despite clearing every Pauli witness. Derivation: section 5.3a.
    """
    sh = unitary_of(compose(["S", "H"]))
    w = su2_components(sh)
    assert w == pytest.approx((0.5, 0.5, -0.5, 0.5), abs=1e-6)

    alpha, _, _ = theorem4_terms(w)
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in alpha.values())
    assert is_forgeable(sh)[0] is True


def test_pauli_witness_free_scheme_is_still_forgeable() -> None:
    """The name is the claim: zero Pauli witnesses, still forgeable.

    Kim Theorem 4 is necessary and sufficient, so this is not a weaker signal
    than the witness search -- it is a stronger one. Ledger V-26.
    """
    spec = load_example("pauli_witness_free_aqs")
    assert spec.forging_witnesses() == ()
    forgeable, triple = spec.kim_forgeable()
    assert forgeable is True
    assert triple is not None


@pytest.mark.parametrize("name", ["baseline_bell_aqs", "choi_fixed_aqs", "pauli_witness_free_aqs"])
def test_every_example_scheme_is_forgeable_under_theorem_4(name: str) -> None:
    """All three bundled schemes are forgeable. None may be presented as secure."""
    assert load_example(name).kim_forgeable()[0] is True


def test_every_clifford_element_is_forgeable() -> None:
    """Conjecture, tested rather than assumed: no Clifford assistant unitary is safe.

    Enumerating all 24 elements of C_1/U(1) and applying Theorem 4 finds no
    counterexample. The consequence is a real constraint on this project's scope:
    a forgery-free assistant unitary is necessarily **non-Clifford**, hence
    outside what Stim can simulate, which is part of why the density-matrix
    engine exists. Ledger V-27.

    If this ever fails, the failing element is a counterexample and is far more
    interesting than the test -- report it, do not adjust the assertion.
    """
    not_forgeable = [g for g in clifford_group() if not is_forgeable(unitary_of(g))[0]]
    assert not_forgeable == [], (
        f"counterexample found: {len(not_forgeable)} Clifford element(s) are not "
        f"forgeable under Theorem 4. This contradicts the enumeration and needs "
        f"investigation before D1 relies on it."
    )


def test_kim_tolerance_is_not_load_bearing() -> None:
    """The Theorem 4 verdict is nowhere near its floating-point tolerance.

    This module evaluates an exact algebraic criterion in floating point, so the
    tolerance has to be shown not to matter. Over the whole Clifford group the
    smallest *non-zero* term is 1/2 -- nearly six orders of magnitude above
    ZERO_TOLERANCE, so no rounding could flip a verdict.
    """
    margin = min(smallest_nonzero_term(unitary_of(g)) for g in clifford_group())
    assert margin == pytest.approx(0.5, abs=1e-6)
    assert margin > ZERO_TOLERANCE * 1e4


def test_kim_operator_is_not_the_t_gate() -> None:
    """Name collision guard. Kim call their operator T; it is a different operator.

    The T gate is diag(1, exp(i*pi/4)). Kim's is a pi rotation about the
    (1,-1,1)/sqrt(3) body diagonal. Both are non-Clifford, for unrelated reasons.
    The registry keeps them under separate names so a spec can never confuse them.
    """
    t_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    assert not np.allclose(W_KIM_FORGERY_FREE, t_gate)

    # Not proportional either: they do not agree even up to a global phase.
    ratio = W_KIM_FORGERY_FREE[0, 0] / t_gate[0, 0]
    assert not np.allclose(W_KIM_FORGERY_FREE, ratio * t_gate)

    assert REGISTRY["W_kim_forgery_free"].is_clifford is False
    assert REGISTRY["T"].is_clifford is False
    assert REGISTRY["W_kim_forgery_free"].tableau is None
    assert "NOT THE T GATE" in REGISTRY["W_kim_forgery_free"].note


def test_kim_operator_is_rejected_as_a_spec_factor() -> None:
    """Non-Clifford means it cannot be a factor, whatever its name."""
    with pytest.raises(NonCliffordOperatorError, match="not a Clifford operator"):
        tableau_of("W_kim_forgery_free")


def test_two_sided_sandwich_raises_rather_than_guessing() -> None:
    """Kim Theorem 4 is stated for one assistant unitary. Rule 1: do not guess.

    A scheme with non-identity factors on both sides has no derived single
    assistant unitary in any source read so far, so the criterion refuses to
    apply itself rather than inventing a convention.
    """
    spec_dict = _spec_dict()
    spec_dict["signing_encryption"] = {
        "family": "uv_type",
        "left_factor": ["H"],
        "right_factor": ["S"],
    }
    spec = load_spec_text(_as_yaml(spec_dict))
    with pytest.raises(NotImplementedError, match="single assistant unitary"):
        spec.kim_forgeable()
