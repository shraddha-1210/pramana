"""The operator vocabulary a scheme specification may name, and the algebra over it.

Every operator Choi discusses lives in the single-qubit Clifford group, and that
group **modulo global phase** is finite: 24 elements. That finiteness is what makes
D1 an exact decision procedure rather than a numerical approximation --
``commutes_up_to_phase(A, B)`` is equality in the quotient group, decided by a
table lookup over bit structures with no floating-point tolerance anywhere. A
tolerance inside a static structural check would undermine the exactness D1 claims.

**Representation.** A single-qubit Clifford modulo global phase is determined by its
conjugation action on the Pauli group -- the pair ``(C X C+, C Z C+)`` of signed
Paulis. That is exactly what ``stim.Tableau`` stores, as bits. Two consequences we
rely on:

* ``A * B == B * A`` in this representation holds **iff** ``AB = lambda BA`` for some
  unit scalar ``lambda``. (``AB = lambda BA`` gives the two the same conjugation
  action; conversely equal conjugation actions differ only by a phase.) So equality
  in the quotient *is* commutation up to global phase, which is the notion Choi's
  attack turns on -- see ``docs/derivations.md`` sections 5.2 and 5.3.
* A non-Clifford operator has no tableau at all. ``T`` is therefore rejected
  structurally rather than by a name check.

**The base set is fixed and not configurable.** The left and right factors sandwich
the four single-qubit Paulis {I, X, Y, Z} and nothing else. Boykin & Roychowdhury's
condition -- a quantum encryption set is optimal iff its unitaries form an
orthonormal basis -- requires exactly four unitaries for one qubit, so exposing the
base set would only admit schemes that are not optimal encryptions. See
``docs/derivations.md`` section 5.3.

Extending the vocabulary means adding a ``RegistryEntry``. Free-form operator
strings are never accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import stim

#: The four single-qubit Paulis the factors sandwich. Fixed, not configurable.
BASE_SET: tuple[str, ...] = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class RegistryEntry:
    """One admissible operator name.

    Attributes:
        name: The token a specification may use.
        tableau: Exact representation as an element of C_1/U(1), or ``None`` when
            the operator is not Clifford and therefore has no tableau.
        is_clifford: Whether the operator lies in the Clifford group. A
            non-Clifford factor takes the protocol outside the stabilizer class,
            where Stim cannot simulate it.
        note: Why a non-Clifford entry exists in the registry at all.
    """

    name: str
    tableau: stim.Tableau | None
    is_clifford: bool
    note: str = ""


def _clifford(name: str) -> RegistryEntry:
    return RegistryEntry(name=name, tableau=stim.Tableau.from_named_gate(name), is_clifford=True)


#: The admissible vocabulary. ``T`` is present so that a spec naming it is *parsed*
#: and then rejected with an explanation, rather than failing later in Layer 2 with
#: an obscure simulator error.
REGISTRY: dict[str, RegistryEntry] = {
    "I": _clifford("I"),
    "X": _clifford("X"),
    "Y": _clifford("Y"),
    "Z": _clifford("Z"),
    "H": _clifford("H"),
    "S": _clifford("S"),
    "W_kim_forgery_free": RegistryEntry(
        name="W_kim_forgery_free",
        tableau=None,
        is_clifford=False,
        note=(
            "(i*sigma_1 - i*sigma_2 + i*sigma_3)/sqrt(3): a pi rotation about the "
            "(1, -1, 1)/sqrt(3) body diagonal, which does not permute the coordinate "
            "axes, so it is non-Clifford and has no tableau. Kim, Lee & Lee "
            "(arXiv:1708.05111) call this operator 'T'. IT IS NOT THE T GATE -- the "
            "T gate is diag(1, exp(i*pi/4)), a different operator. The bare name 'T' "
            "is reserved for the gate, so this one is registered under its own name "
            "to make the collision impossible. It is the only known assistant unitary "
            "here for which no forgeable message exists (Kim Theorem 4), and it is "
            "outside the class Stim can simulate."
        ),
    ),
    "T": RegistryEntry(
        name="T",
        tableau=None,
        is_clifford=False,
        note=(
            "T = diag(1, exp(i*pi/4)) is non-Clifford: it has no stabilizer tableau, "
            "so Stim cannot simulate a protocol containing it. Admitting a T factor "
            "would require stabilizer-rank simulation, which this project documents "
            "and deliberately does not build."
        ),
    ),
}

CLIFFORD_NAMES: tuple[str, ...] = tuple(n for n, e in REGISTRY.items() if e.is_clifford)


class NonCliffordOperatorError(ValueError):
    """Raised when a specification names an operator outside the Clifford group."""


def tableau_of(name: str) -> stim.Tableau:
    """Return the exact representation of a registry name.

    Raises:
        KeyError: The name is not in the registry. Free-form strings are never
            accepted; extend ``REGISTRY`` instead.
        NonCliffordOperatorError: The name is registered but not Clifford.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown operator {name!r}. Admissible operators are "
            f"{', '.join(REGISTRY)}. Operator names are not free-form: extend "
            f"pramana.spec.operators.REGISTRY to add one."
        )
    entry = REGISTRY[name]
    if entry.tableau is None:
        raise NonCliffordOperatorError(f"{name} is not a Clifford operator. {entry.note}")
    return entry.tableau


def compose(names: list[str] | tuple[str, ...]) -> stim.Tableau:
    """Compose a factor sequence, left to right as written.

    ``["S", "H"]`` denotes the product S*H -- S applied to the left of H, matching
    how ``SH`` is written on paper.

    The composition order is not assumed from documentation. ``stim.Tableau``'s
    ``A * B`` was verified at runtime to correspond to the matrix product ``A @ B``
    (ledger V-20), and ``test_factor_sequence_composition_order`` re-proves it.

    An empty sequence composes to the identity.
    """
    product = stim.Tableau(1)
    for name in names:
        product = product * tableau_of(name)
    return product


def commutes_up_to_phase(a: stim.Tableau, b: stim.Tableau) -> bool:
    """Whether ``AB = lambda BA`` for some unit scalar ``lambda``.

    Exact: equality of bit structures in C_1/U(1), no tolerance. Global phase is
    physically unobservable, which is precisely why Choi's attack works -- see
    ``docs/derivations.md`` section 5.2.
    """
    return bool(a * b == b * a)


@lru_cache(maxsize=1)
def clifford_group() -> tuple[stim.Tableau, ...]:
    """Every element of C_1/U(1), generated by closure from {H, S}.

    The order is *generated*, never asserted into existence: if the representation
    were wrong the closure would return a different count, and
    ``test_clifford_group_has_order_24`` would fail rather than be adjusted.
    """
    identity = stim.Tableau(1)
    generators = [tableau_of("H"), tableau_of("S")]
    # stim.Tableau is not hashable; its string form is a faithful canonical key.
    found: dict[str, stim.Tableau] = {str(identity): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for element in frontier:
            for g in generators:
                for product in (element * g, g * element):
                    if str(product) not in found:
                        found[str(product)] = product
                        nxt.append(product)
        frontier = nxt
    return tuple(found.values())


def encryption_set(
    left_factor: list[str] | tuple[str, ...],
    right_factor: list[str] | tuple[str, ...],
) -> tuple[stim.Tableau, ...]:
    """The operator set ``U {I, X, Y, Z} V`` induced by a pair of factor sequences.

    This is Choi's (U,V)-type quantum encryption, ``docs/derivations.md`` section 5.3.
    """
    u = compose(left_factor)
    v = compose(right_factor)
    return tuple(u * tableau_of(p) * v for p in BASE_SET)


def forging_witnesses(
    encryption: tuple[stim.Tableau, ...],
    rotation: tuple[stim.Tableau, ...],
) -> tuple[stim.Tableau, ...]:
    """Every non-trivial ``Q`` that commutes up to phase with all of ``E`` and ``R``.

    This is the D1 predicate from ``docs/derivations.md`` section 5.3, exactly as
    written there:

        A scheme is vulnerable to Choi-class existential forgery if and only if
        there exists a non-trivial operator Q that commutes, up to global phase,
        with every operator in the signing encryption set E and every operator in
        the random rotation set R.

    The search runs over the whole 24-element quotient group, not only the Paulis,
    so a non-Pauli witness would be found if one existed. Restricting the *reported
    attack class* to Pauli witnesses is a separate scope decision (ledger V-09).

    D1 the detector -- its verdict object, evidence, repair suggestion and the
    sub-checks D1c and D1d -- is Layer 6. This function is the algebra the schema's
    meaning depends on, which is why it is validated here at Layer 1.

    Returns:
        The witnesses, in the generation order of ``clifford_group()``. Empty means
        not vulnerable *to this attack class*, which is not the same as secure.
    """
    identity = str(stim.Tableau(1))
    operators = list(encryption) + list(rotation)
    return tuple(
        q
        for q in clifford_group()
        if str(q) != identity and all(commutes_up_to_phase(q, op) for op in operators)
    )


def name_of(tableau: stim.Tableau) -> str:
    """Best-effort registry name for a tableau, for evidence strings.

    Returns the registry name when the element is one of the named operators, and
    a stim repr otherwise -- a composite has no single name.
    """
    for entry in REGISTRY.values():
        if entry.tableau is not None and entry.tableau == tableau:
            return entry.name
    return repr(tableau)
