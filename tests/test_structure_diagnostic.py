"""Tests for `BaseConverter.structure_diagnostic`, the diagnostic union
structuring entry point.

Run locally with:

    python -m pytest tests/test_structure_diagnostic.py -q
"""

from typing import Optional, Union

import pytest
from attrs import define

import cattrs
from cattrs import (
    BaseConverter,
    Converter,
    UnionAmbiguityError,
    UnionStructureError,
    transform_error,
)
from cattrs.errors import (
    AttributeValidationNote,
    ClassValidationError,
    IterableValidationError,
    IterableValidationNote,
)


@define
class A:
    x: int


@define
class B:
    y: str


@define
class C:
    xs: list[int]


@define
class D:
    d: str


@pytest.fixture(params=[BaseConverter, Converter], scope="function")
def diag_converter(request) -> BaseConverter:
    """Both converter classes expose the diagnostic entry point."""
    return request.param()


class TestBranchDiagnostics:
    """Every branch is tried once and its nested errors are preserved."""

    def test_all_branches_reported_in_order(self, diag_converter):
        """A{x: int} fails on x, B{y: str} fails on missing y; both are shown."""
        with pytest.raises(UnionStructureError) as exc_info:
            diag_converter.structure_diagnostic({"x": "bad"}, Union[A, B])

        exc = exc_info.value
        assert exc.cl == Union[A, B]
        assert exc.branches == (A, B)
        assert len(exc.exceptions) == 2

        def unwrap(e):
            # `Converter` wraps attribute errors, `BaseConverter` does not.
            return e.exceptions[0] if isinstance(e, ClassValidationError) else e

        # Branch A: `x` failed to structure.
        assert isinstance(unwrap(exc.exceptions[0]), ValueError)

        # Branch B: `y` is missing (KeyError with detailed validation,
        # a plain TypeError from `B.__init__` without it).
        err_b = unwrap(exc.exceptions[1])
        assert isinstance(err_b, (KeyError, TypeError))
        assert "y" in str(err_b)

    def test_detailed_nested_field_errors(self):
        """With detailed validation, branch errors keep attribute notes."""
        conv = Converter()
        with pytest.raises(UnionStructureError) as exc_info:
            conv.structure_diagnostic({"x": "bad"}, Union[A, B])

        err_a, err_b = exc_info.value.exceptions

        assert isinstance(err_a, ClassValidationError)
        assert err_a.cl is A
        notes = [
            n
            for n in err_a.exceptions[0].__notes__
            if isinstance(n, AttributeValidationNote)
        ]
        assert [n.name for n in notes] == ["x"]
        assert isinstance(err_a.exceptions[0], ValueError)

        assert isinstance(err_b, ClassValidationError)
        assert err_b.cl is B
        assert isinstance(err_b.exceptions[0], KeyError)
        assert err_b.exceptions[0].args[0] == "y"

    def test_not_just_last_branch(self, diag_converter):
        """The first branch error is not discarded."""
        with pytest.raises(UnionStructureError) as exc_info:
            diag_converter.structure_diagnostic({"x": "bad"}, Union[A, B])
        exc = exc_info.value
        assert exc.branches == (A, B)
        assert len(exc.exceptions) == 2

    def test_nested_list_paths_preserved(self):
        """Nested iterable errors keep their indices (paths) inside the report."""
        conv = Converter()
        with pytest.raises(UnionStructureError) as exc_info:
            conv.structure_diagnostic({"xs": [1, "bad"]}, Union[C, D])

        err_c = exc_info.value.exceptions[0]
        assert isinstance(err_c, ClassValidationError)
        (iter_err,) = err_c.exceptions
        assert isinstance(iter_err, IterableValidationError)
        notes = [
            n
            for n in iter_err.exceptions[0].__notes__
            if isinstance(n, IterableValidationNote)
        ]
        assert [n.index for n in notes] == [1]

        # The nested errors survive `transform_error` with full paths.
        lines = transform_error(err_c)
        assert any("$.xs[1]" in line for line in lines)

    def test_optional_none_branch(self, diag_converter):
        """`None` matches the NoneType branch of an Optional."""
        assert diag_converter.structure_diagnostic(None, Optional[A]) is None

        with pytest.raises(UnionStructureError) as exc_info:
            diag_converter.structure_diagnostic({"x": "bad"}, Optional[A])
        exc = exc_info.value
        assert len(exc.exceptions) == 2
        assert isinstance(exc.exceptions[1], ValueError)  # NoneType branch

    def test_non_union_passthrough(self, diag_converter):
        """Non-union targets behave exactly like `structure`."""
        assert diag_converter.structure_diagnostic({"x": 1}, A) == A(1)
        with pytest.raises((ClassValidationError, ValueError)):
            diag_converter.structure_diagnostic({"x": "bad"}, A)


class TestSuccessAndAmbiguity:
    def test_single_success_returns_value(self, diag_converter):
        """A single matching branch returns the plain structured value."""
        res = diag_converter.structure_diagnostic({"x": 1}, Union[A, B])
        assert res == A(1)
        assert type(res) is A

    def test_strict_ambiguity_reports_all_successes(self, diag_converter):
        """Strict mode reports both succeeding branches."""
        with pytest.raises(UnionAmbiguityError) as exc_info:
            diag_converter.structure_diagnostic({"x": 1, "y": "a"}, Union[A, B])

        exc = exc_info.value
        assert exc.cl == Union[A, B]
        assert [b for b, _ in exc.successes] == [A, B]
        assert [v for _, v in exc.successes] == [A(1), B("a")]
        assert "A" in str(exc) and "B" in str(exc)

    def test_non_strict_ambiguity_returns_first(self, diag_converter):
        """Non-strict mode resolves to the first branch in registration order."""
        res = diag_converter.structure_diagnostic(
            {"x": 1, "y": "a"}, Union[A, B], strict_ambiguity=False
        )
        assert res == A(1)

        res = diag_converter.structure_diagnostic(
            {"x": 1, "y": "a"}, Union[B, A], strict_ambiguity=False
        )
        assert res == B("a")


class TestHookPriorityAndCallCounts:
    def test_custom_branch_hook_used_and_called_once(self, diag_converter):
        """Custom hooks keep their priority and run at most once per branch."""
        calls = []

        def hook_a(obj, cl):
            calls.append((obj, cl))
            return A(999)

        diag_converter.register_structure_hook(A, hook_a)

        res = diag_converter.structure_diagnostic({"x": 1}, Union[A, B])
        assert res == A(999)  # The custom hook result is used.
        assert len(calls) == 1  # Called exactly once, even on the success path.

    def test_custom_hook_called_once_on_failure(self, diag_converter):
        """A failing custom hook is not retried."""
        calls = []

        def hook_a(obj, cl):
            calls.append(obj)
            raise ValueError("nope")

        diag_converter.register_structure_hook(A, hook_a)

        with pytest.raises(UnionStructureError) as exc_info:
            diag_converter.structure_diagnostic({}, Union[A, B])
        assert len(calls) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)

    def test_union_level_hook_takes_priority_and_called_once(self, diag_converter):
        """A hook registered for the union itself wins, as with `structure`."""
        calls = []

        def union_hook(obj, cl):
            calls.append(obj)
            return "custom"

        diag_converter.register_structure_hook(Union[A, B], union_hook)

        assert diag_converter.structure_diagnostic({"x": 1}, Union[A, B]) == "custom"
        assert len(calls) == 1

    def test_no_partial_results_exposed(self, diag_converter):
        """Failures raise; no half-structured values leak to the caller."""
        try:
            diag_converter.structure_diagnostic({"x": "bad"}, Union[A, B])
        except UnionStructureError as exc:
            assert not hasattr(exc, "value")
            assert all(isinstance(e, Exception) for e in exc.exceptions)
        else:
            pytest.fail("expected UnionStructureError")


class TestDefaultCompatibility:
    """The default `structure` behavior is unchanged."""

    @pytest.mark.parametrize("detailed_validation", [True, False])
    def test_diagnostic_works_without_detailed_validation(
        self, detailed_validation: bool
    ):
        conv = Converter(detailed_validation=detailed_validation)
        with pytest.raises(UnionStructureError) as exc_info:
            conv.structure_diagnostic({"x": "bad"}, Union[A, B])
        assert len(exc_info.value.exceptions) == 2

    def test_default_structure_unchanged(self):
        conv = Converter()
        # The default union strategy still uses disambiguation and raises a
        # single ClassValidationError for the chosen class.
        with pytest.raises(ClassValidationError):
            conv.structure({"x": "bad"}, Union[A, B])
        assert conv.structure({"x": 1}, Union[A, B]) == A(1)

    def test_module_level_entry_point(self):
        assert cattrs.structure_diagnostic({"x": 1}, Union[A, B]) == A(1)
        with pytest.raises(UnionAmbiguityError):
            cattrs.structure_diagnostic({"x": 1, "y": "a"}, Union[A, B])
