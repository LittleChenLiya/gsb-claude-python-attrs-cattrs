"""Tests for the diagnostic union structuring entry point."""

from typing import Optional, Union

import pytest
from attrs import define

from cattrs import (
    AmbiguousUnionStructureError,
    BaseConverter,
    Converter,
    UnionValidationError,
    transform_error,
)


@define
class A:
    x: int


@define
class B:
    y: str


@define
class C:
    items: list[int]


@define
class D:
    name: str


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_all_branches_fail(cls):
    """Every branch error is reported, in registration order."""
    c = cls()
    with pytest.raises(UnionValidationError) as exc_info:
        c.structure_diagnostic({"x": "bad"}, Union[A, B])

    exc = exc_info.value
    assert exc.cl is Union[A, B]
    assert len(exc.exceptions) == 2
    # Registration order: A then B.
    notes = [
        next(
            n
            for n in e.__notes__
            if n.__class__.__name__ == "UnionMemberValidationNote"
        )
        for e in exc.exceptions
    ]
    assert [n.type for n in notes] == [A, B]


def test_diagnostic_shows_field_errors():
    """The acceptance case: x conversion failure and y missing both show."""
    c = Converter()
    with pytest.raises(UnionValidationError) as exc_info:
        c.structure_diagnostic({"x": "bad"}, Union[A, B])

    lines = transform_error(exc_info.value)
    assert any("$.A.x" in line and "int" in line for line in lines)
    assert any("$.B.y" in line and "missing" in line for line in lines)


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_returns_first_success(cls):
    """Without strict mode, the first successful branch wins."""
    c = cls()
    res = c.structure_diagnostic({"x": 1, "y": "s"}, Union[A, B])
    assert res == A(1)


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_strict_ambiguity(cls):
    """Strict mode reports when more than one branch succeeds."""
    c = cls()
    with pytest.raises(AmbiguousUnionStructureError) as exc_info:
        c.structure_diagnostic({"x": 1, "y": "s"}, Union[A, B], strict=True)

    exc = exc_info.value
    assert exc.union is Union[A, B]
    assert [member for member, _ in exc.successes] == [A, B]
    assert [result for _, result in exc.successes] == [A(1), B("s")]


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_single_success(cls):
    """A single successful branch returns its value."""
    c = cls()
    assert c.structure_diagnostic({"x": 1}, Union[A, B]) == A(1)
    assert c.structure_diagnostic({"y": "s"}, Union[A, B]) == B("s")


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_optional_none(cls):
    """Optional unions still handle None."""
    c = cls()
    assert c.structure_diagnostic(None, Optional[A]) is None
    assert c.structure_diagnostic({"x": 1}, Optional[A]) == A(1)


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_non_union_delegates(cls):
    """Non-union targets fall back to normal structuring."""
    c = cls()
    assert c.structure_diagnostic({"x": 1}, A) == A(1)
    assert c.structure_diagnostic(1, int) == 1


def test_diagnostic_hook_called_at_most_once():
    """Each branch's structure hook is invoked at most once."""
    c = Converter()
    calls = []

    def hook(obj, cl):
        calls.append(cl)
        if cl is A:
            return A(int(obj["x"]))  # int("bad") raises
        return B(obj["y"])  # missing y raises KeyError

    c.register_structure_hook(A, hook)
    c.register_structure_hook(B, hook)

    c.structure_diagnostic({"x": 1, "y": "s"}, Union[A, B])
    assert calls == [A, B]

    calls.clear()
    with pytest.raises(UnionValidationError):
        c.structure_diagnostic({"x": "bad"}, Union[A, B])
    assert calls == [A, B]


def test_diagnostic_custom_hook_priority():
    """Custom hooks take precedence over generated ones."""
    c = Converter()
    c.register_structure_hook(A, lambda obj, _: A(obj["x"] + 100))

    res = c.structure_diagnostic({"x": 1}, Union[A, B])
    assert res == A(101)


def test_diagnostic_hook_factory_priority():
    """Hook factories are consulted through the normal dispatch."""
    c = Converter()
    c.register_structure_hook_factory(
        lambda t: t is A, lambda t: lambda obj, _: A(obj["x"] + 100)
    )

    res = c.structure_diagnostic({"x": 1}, Union[A, B])
    assert res == A(101)


def test_diagnostic_list_nested_paths():
    """Errors nested in list fields keep their index paths."""
    c = Converter()
    with pytest.raises(UnionValidationError) as exc_info:
        c.structure_diagnostic({"items": ["bad"]}, Union[C, D])

    lines = transform_error(exc_info.value)
    assert any("$.C.items[0]" in line for line in lines)
    assert any("$.D.name" in line for line in lines)


def test_diagnostic_sequence_of_unions():
    """Lists of unions are diagnosed per element, with index paths."""
    c = Converter()
    with pytest.raises(Exception) as exc_info:
        c.structure_diagnostic([{"x": "bad"}], list[Union[A, B]])

    lines = transform_error(exc_info.value)
    assert any("$[0].A.x" in line and "int" in line for line in lines)
    assert any("$[0].B.y" in line and "missing" in line for line in lines)


def test_diagnostic_sequence_of_unions_success():
    """Successful lists of unions return the structured elements."""
    c = Converter()
    res = c.structure_diagnostic([{"x": 1}, {"y": "s"}], list[Union[A, B]])
    assert res == [A(1), B("s")]


def test_diagnostic_sequence_of_unions_strict():
    """Strict ambiguity is reported per element."""
    c = Converter()
    with pytest.raises(AmbiguousUnionStructureError):
        c.structure_diagnostic([{"x": 1, "y": "s"}], list[Union[A, B]], strict=True)


@pytest.mark.parametrize("cls", (BaseConverter, Converter))
def test_diagnostic_default_structure_unchanged(cls):
    """The default structure entry point is unaffected."""
    c = cls()
    # Default structure still uses disambiguation, so it fails with only the
    # disambiguated branch's error (never branch B's).
    with pytest.raises(Exception) as exc_info:
        c.structure({"x": "bad"}, Union[A, B])
    lines = transform_error(exc_info.value)
    assert not any("$.B" in line for line in lines)
    # The diagnostic entry is separate and reports every branch.
    with pytest.raises(UnionValidationError) as diag_exc:
        c.structure_diagnostic({"x": "bad"}, Union[A, B])
    assert len(diag_exc.value.exceptions) == 2


def test_diagnostic_no_partial_result_exposed():
    """A failed branch's partially structured result is never returned."""
    c = Converter()

    @define
    class Partial:
        x: int

    def bad_hook(obj, cl):
        # Simulate a hook that mutates state before failing.
        raise ValueError("boom")

    c.register_structure_hook(Partial, bad_hook)
    with pytest.raises(UnionValidationError):
        c.structure_diagnostic({"x": 1}, Union[Partial, B])
