"""Tests for `cattrs.tree`, the read-only error organizing layer.

Run just these tests with:

    python -m pytest tests/test_error_tree.py -v
"""

import pytest
from attrs import define
from attrs.exceptions import FrozenInstanceError

from cattrs import Converter
from cattrs.errors import (
    ClassValidationError,
    IterableValidationError,
    IterableValidationNote,
)
from cattrs.tree import (
    CycleNode,
    ErrorTree,
    Node,
    Segment,
    build_error_tree,
    format_path,
    safe_format_exception,
)


def _structure_error(obj, cl):
    """Structure with detailed validation, returning the raised error."""
    try:
        Converter(detailed_validation=True).structure(obj, cl)
    except (ClassValidationError, IterableValidationError) as exc:
        return exc
    raise AssertionError("structuring did not fail")


def _notes_of(exc):
    return [getattr(e, "__notes__", None) for e in exc.exceptions]


class TestNestedListTruncation:
    """Nested lists with five errors, capped at two kept leaves."""

    @pytest.fixture()
    def exc(self):
        return _structure_error([["a"], ["b"], ["c"], ["d"], ["e"]], list[list[int]])

    def test_total_preserved_but_leaves_capped(self, exc):
        tree = build_error_tree(exc, max_leaves=2)

        assert isinstance(tree, ErrorTree)
        assert tree.total == 5
        assert tree.kept == 2
        assert tree.dropped == 3

    def test_first_two_paths_are_real_and_accurate(self, exc):
        tree = build_error_tree(exc, max_leaves=2)

        leaves = list(tree.leaves())
        assert [leaf.path_str for leaf in leaves] == ["$[0][0]", "$[1][0]"]
        # The real indices are kept as ints, not renumbered strings.
        assert leaves[0].path == (Segment("index", 0), Segment("index", 0))
        assert leaves[1].path == (Segment("index", 1), Segment("index", 0))
        assert leaves[0].message == "invalid value for type, expected int"

    def test_uncapped_keeps_everything_in_order(self, exc):
        tree = build_error_tree(exc)

        assert tree.total == 5
        assert tree.kept == 5
        assert not tree.truncated
        assert tree.is_complete
        assert [leaf.path_str for leaf in tree.leaves()] == [
            f"$[{i}][0]" for i in range(5)
        ]

    def test_truncation_is_an_independent_marker(self, exc):
        tree = build_error_tree(exc, max_leaves=2)

        # Truncation is marked independently of the kept leaves...
        assert tree.truncated
        assert not tree.is_complete
        # ...and a truncated tree can never look like it fully passed.
        assert not tree.ok
        # Dropped counts are available per-node, not just globally.
        assert tree.root.dropped == 3
        dropped_by_child = {
            seg.value: child.dropped
            for seg, child in tree.root.children
            if isinstance(child, Node)
        }
        assert dropped_by_child == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}

    def test_max_leaves_zero_still_counts_and_marks(self, exc):
        tree = build_error_tree(exc, max_leaves=0)

        assert tree.total == 5
        assert tree.kept == 0
        assert tree.truncated
        assert not tree.ok  # No leaves kept, but clearly not a pass.

    def test_original_exception_notes_are_unchanged(self, exc):
        before = _notes_of(exc)
        inner_before = [
            _notes_of(e) for e in exc.exceptions if hasattr(e, "exceptions")
        ]

        build_error_tree(exc, max_leaves=2)
        build_error_tree(exc)

        assert _notes_of(exc) == before
        assert [
            _notes_of(e) for e in exc.exceptions if hasattr(e, "exceptions")
        ] == inner_before
        # The notes are still the original IterableValidationNotes.
        for sub in exc.exceptions:
            for note in getattr(sub, "__notes__", []):
                assert note.__class__ is IterableValidationNote


class TestClassesAndKeys:
    """Real field names and mapping keys, not reordered by truncation."""

    def test_class_fields_and_mapping_keys(self):
        @define
        class Inner:
            a: int

        @define
        class Outer:
            x: Inner
            y: dict[str, int]

        exc = _structure_error(
            {"x": {"a": "bad"}, "y": {"k": "bad", "z": "also bad"}}, Outer
        )
        tree = build_error_tree(exc, max_leaves=2)

        assert tree.total == 3
        assert tree.truncated
        assert [leaf.path_str for leaf in tree.leaves()] == ["$.x.a", "$.y['k']"]
        leaf = next(iter(tree.leaves()))
        assert leaf.path == (Segment("attribute", "x"), Segment("attribute", "a"))

    def test_format_path(self):
        segments = (
            Segment("attribute", "field"),
            Segment("index", 3),
            Segment("key", "k"),
        )
        assert format_path(segments) == "$.field[3]['k']"
        assert format_path(()) == "$"


class TestCauses:
    """Causes keep their types, never their input values."""

    def test_cause_types_kept_without_values(self):
        try:
            try:
                {}["missing"]
            except KeyError as ke:
                raise ValueError("the input value was 12345-secret") from ke
        except ValueError as ve:
            tree = build_error_tree(ve)

        (leaf,) = tree.leaves()
        assert leaf.cause_types == ("ValueError", "KeyError")
        assert not leaf.cause_cycle
        # Neither the message nor the cause chain leaks the input value.
        assert "12345-secret" not in leaf.message
        assert "12345-secret" not in repr(leaf)

    def test_safe_format_exception_never_interpolates_instance(self):
        class WeirdError(Exception):
            pass

        exc = WeirdError("sensitive-input-value")
        assert safe_format_exception(exc, None) == "unknown error (WeirdError)"
        assert "sensitive-input-value" not in safe_format_exception(exc, None)

        te = TypeError("sensitive-input-value is broken")
        assert safe_format_exception(te, None) == "invalid type (TypeError)"

    def test_cyclic_cause_chain_stops_explicitly(self):
        a = ValueError("a")
        b = TypeError("b")
        a.__cause__ = b
        b.__cause__ = a  # A cycle, only constructible by assignment.

        tree = build_error_tree(a)

        (leaf,) = tree.leaves()
        assert leaf.cause_types == ("ValueError", "TypeError")
        assert leaf.cause_cycle  # Stopped explicitly, flagged on the leaf.
        assert tree.total == 1

    def test_longer_cause_chain(self):
        c = ZeroDivisionError("c")
        b = TypeError("b")
        b.__cause__ = c
        a = ValueError("a")
        a.__cause__ = b

        (leaf,) = build_error_tree(a).leaves()
        assert leaf.cause_types == ("ValueError", "TypeError", "ZeroDivisionError")
        assert not leaf.cause_cycle


class TestGroupCycles:
    """Cyclic exception groups are stopped explicitly with markers."""

    def test_cyclic_group_chain(self):
        # `ExceptionGroup.exceptions` is read-only, so a cycle is simulated
        # with a subclass whose `exceptions` point back to an ancestor.
        class BackToTopError(IterableValidationError):
            @property
            def exceptions(self):
                return (top,)

        back = BackToTopError("back", [ValueError("placeholder")], list)
        back.add_note(IterableValidationNote("while structuring element 0", 0, list))
        top = IterableValidationError("top", [back], list)
        top.add_note(IterableValidationNote("while structuring element 1", 1, list))

        # top -> [0] -> back -> [1] -> top -> ... would recurse forever
        # without an explicit stop.
        tree = build_error_tree(top)

        assert tree.total == 0
        assert not tree.truncated
        # The cycle was stopped explicitly with a marker node.
        ((seg, child),) = tree.root.children
        assert seg == Segment("index", 0)
        assert isinstance(child, Node)
        ((seg2, marker),) = child.children
        assert seg2 == Segment("index", 1)
        assert isinstance(marker, CycleNode)
        assert marker.exc_type == "IterableValidationError"


class TestReadOnly:
    """The layer only reads; it never re-runs structuring."""

    def test_no_converter_involved_and_repeatable(self):
        exc = _structure_error([["a"], ["b"]], list[list[int]])

        first = build_error_tree(exc, max_leaves=1)
        second = build_error_tree(exc, max_leaves=1)

        # Purely a function of the exception: equal, immutable results.
        assert first == second
        assert first.total == 2
        with pytest.raises(FrozenInstanceError):
            first.total = 5  # Frozen; the tree is read-only.

    def test_plain_exception_at_root(self):
        tree = build_error_tree(ValueError("nope"))

        assert tree.total == 1
        assert not tree.truncated
        assert tree.is_complete
        (leaf,) = tree.leaves()
        assert leaf.path == ()
        assert leaf.path_str == "$"
        assert leaf.message == "invalid value"
        assert leaf.cause_types == ("ValueError",)

    def test_negative_max_leaves_rejected(self):
        with pytest.raises(ValueError):
            build_error_tree(ValueError("x"), max_leaves=-1)

    def test_exceptions_without_notes_attach_at_current_path(self):
        # A group whose children carry no notes at all.
        group = IterableValidationError("g", [ValueError("x")], list)
        for sub in group.exceptions:
            assert not hasattr(sub, "__notes__")

        tree = build_error_tree(group)
        (leaf,) = tree.leaves()
        assert leaf.path == ()
        assert leaf.message == "invalid value"
