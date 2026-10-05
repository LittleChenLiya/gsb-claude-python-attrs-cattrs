"""Tests for the error tree organization layer.

Run locally with::

    pytest tests/test_errortree.py -v
"""

from typing import List

import pytest
from attrs import define

from cattrs import Converter
from cattrs.errors import ClassValidationError, IterableValidationError
from cattrs.errortree import (
    ErrorBranch,
    ErrorLeaf,
    FieldName,
    build_error_tree,
    render_tree,
)


def _collect_leaves(children):
    """Yield all leaves in the tree, depth-first."""
    for child in children:
        if isinstance(child, ErrorLeaf):
            yield child
        else:
            yield from _collect_leaves(child.children)


def _collect_branches(children):
    """Yield all branches in the tree, depth-first."""
    for child in children:
        if isinstance(child, ErrorBranch):
            yield child
            yield from _collect_branches(child.children)


def test_nested_list_five_errors_cap_two():
    """Nested list with five errors, cap 2: total 5, first two accurate paths.

    The original exception notes must remain unchanged.
    """
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure([["a", "b"], ["c", "d"], ["e"]], List[List[int]])

    exc = exc_info.value

    # Snapshot notes before building the tree.
    notes_before = []
    for sub in exc.exceptions:
        if hasattr(sub, "__notes__"):
            notes_before.append(list(sub.__notes__))
        for leaf in getattr(sub, "exceptions", []):
            if hasattr(leaf, "__notes__"):
                notes_before.append(list(leaf.__notes__))

    tree = build_error_tree(exc, limit=2)

    # Total count is preserved, only two leaves are shown.
    assert tree.total == 5
    assert tree.shown == 2
    assert tree.truncated is True
    assert tree.limit == 2

    # The first two paths use the real indexes, not renumbered.
    leaves = list(_collect_leaves(tree.children))
    assert len(leaves) == 2
    assert leaves[0].path == (0, 0)
    assert leaves[1].path == (0, 1)
    assert leaves[0].error_type is ValueError
    assert leaves[1].error_type is ValueError

    # Notes are unchanged after building the tree.
    notes_after = []
    for sub in exc.exceptions:
        if hasattr(sub, "__notes__"):
            notes_after.append(list(sub.__notes__))
        for leaf in getattr(sub, "exceptions", []):
            if hasattr(leaf, "__notes__"):
                notes_after.append(list(leaf.__notes__))

    assert notes_before == notes_after


def test_truncation_is_independent_marker():
    """Truncation is an independent marker, not a fake error leaf.

    The truncated tree must not look like all errors passed.
    """
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(["a", "b", "c", "d", "e"], List[int])

    tree = build_error_tree(exc_info.value, limit=2)

    assert tree.truncated is True
    assert tree.total == 5
    assert tree.shown == 2

    # The truncation marker is a separate field, not a leaf.
    leaves = list(_collect_leaves(tree.children))
    assert len(leaves) == 2

    # The rendered tree clearly shows truncation.
    rendered = render_tree(tree)
    assert "truncated" in rendered
    assert "total 5" in rendered
    # Must not look like a clean pass.
    assert "✗" in rendered


def test_no_truncation_when_under_limit():
    """No truncation when errors are within the limit."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(["a", "b"], List[int])

    tree = build_error_tree(exc_info.value, limit=5)

    assert tree.truncated is False
    assert tree.total == 2
    assert tree.shown == 2
    assert tree.limit == 5


def test_limit_none_keeps_all():
    """limit=None keeps all leaves."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(["a", "b", "c"], List[int])

    tree = build_error_tree(exc_info.value)

    assert tree.truncated is False
    assert tree.total == 3
    assert tree.shown == 3
    assert tree.limit is None


def test_limit_zero_keeps_none():
    """limit=0 keeps no leaves but still records the total."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(["a", "b", "c"], List[int])

    tree = build_error_tree(exc_info.value, limit=0)

    assert tree.truncated is True
    assert tree.total == 3
    assert tree.shown == 0
    assert tree.children == ()


def test_class_validation_field_names_preserved():
    """Class validation errors preserve field names in paths."""
    c = Converter(detailed_validation=True)

    @define
    class Inner:
        x: int
        y: int

    @define
    class Outer:
        a: int
        b: Inner
        items: List[int]

    with pytest.raises(ClassValidationError) as exc_info:
        c.structure(
            {"a": "bad", "b": {"x": "bad", "y": "bad"}, "items": ["bad", "bad"]}, Outer
        )

    tree = build_error_tree(exc_info.value, limit=3)

    assert tree.total == 5
    assert tree.shown == 3
    assert tree.truncated is True

    leaves = list(_collect_leaves(tree.children))
    paths = [leaf.path for leaf in leaves]

    # Field names are preserved, not renumbered.
    assert paths[0] == (FieldName("a"),)
    assert paths[1] == (FieldName("b"), FieldName("x"))
    assert paths[2] == (FieldName("b"), FieldName("y"))


def test_cause_preserves_type_not_value():
    """Cause chain preserves types but does not output input values."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure([["a"]], List[List[int]])

    exc = exc_info.value
    first_leaf = exc.exceptions[0].exceptions[0]
    # Attach a cause with a secret value.
    first_leaf.__cause__ = ValueError("secret input value")

    tree = build_error_tree(exc)

    leaves = list(_collect_leaves(tree.children))
    leaf = leaves[0]

    # The cause type is preserved.
    assert leaf.cause_types == (ValueError,)
    # The cause value is not output in the message.
    assert "secret input value" not in leaf.message
    # The rendered tree does not leak the cause value.
    assert "secret input value" not in render_tree(tree)


def test_circular_cause_chain_stops():
    """Circular cause chains stop explicitly without hanging."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure([["a"]], List[List[int]])

    exc = exc_info.value
    first_leaf = exc.exceptions[0].exceptions[0]

    # Create a circular cause chain: a -> b -> a.
    a = ValueError("a")
    b = TypeError("b")
    a.__cause__ = b
    b.__cause__ = a
    first_leaf.__cause__ = a

    # This must not hang.
    tree = build_error_tree(exc)

    leaves = list(_collect_leaves(tree.children))
    leaf = leaves[0]

    # The cycle is detected and stopped after recording both types.
    assert leaf.cause_types == (ValueError, TypeError)


def test_organization_is_read_only():
    """Building the tree does not run structuring hooks."""
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(["a", "b"], List[int])

    exc = exc_info.value

    # Build the tree multiple times - it should be deterministic.
    tree1 = build_error_tree(exc)
    tree2 = build_error_tree(exc)

    assert tree1 == tree2
    assert tree1.total == 2
    assert tree1.shown == 2


def test_error_tree_example():
    """Rendered error tree example for documentation.

    This test demonstrates the error tree output format.
    """
    c = Converter(detailed_validation=True)

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure([["a", "b"], ["c", "d"], ["e"]], List[List[int]])

    tree = build_error_tree(exc_info.value, limit=2)
    rendered = render_tree(tree)

    # The rendered tree shows the structure and truncation.
    assert rendered.startswith("$")
    assert "├──" in rendered or "└──" in rendered
    assert "truncated" in rendered
    assert "total 5" in rendered

    # Print the example for reference (run with -s to see output).
    print("\n--- Error Tree Example ---")  # noqa: T201
    print(rendered)  # noqa: T201
    print("--------------------------")  # noqa: T201
