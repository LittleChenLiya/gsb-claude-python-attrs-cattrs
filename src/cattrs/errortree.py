"""Error organization into path trees.

This module provides an explicit error organization layer that turns
:class:`IterableValidationError` and :class:`ClassValidationError` exception
groups into a path tree. The tree mirrors the nested structure of the
exceptions and supports truncating the number of leaf errors while keeping
the total count.

The organization is read-only: it only reads the exception tree and never
runs structuring hooks. The original exception (and its notes) is left
untouched.

.. versionadded:: 24.1.0
"""

from typing import Any, Callable, Optional, Union

from attrs import frozen

from .errors import (
    AttributeValidationNote,
    BaseValidationError,
    ClassValidationError,
    IterableValidationError,
    IterableValidationNote,
)
from .v import format_exception

__all__ = [
    "ErrorBranch",
    "ErrorLeaf",
    "ErrorTree",
    "FieldName",
    "build_error_tree",
    "render_tree",
]

# A path component: field name, list index, or dict key.
_PathComponent = Any
# A tuple of path components from the root to a leaf.
_Path = tuple[_PathComponent, ...]
# A raw leaf collected during traversal: (path, exception, expected_type).
_RawLeaf = tuple[_Path, BaseException, Any]


@frozen
class FieldName:
    """A path component representing a field name.

    Field names come from :class:`AttributeValidationNote` (class fields)
    and are rendered as ``.name`` in paths, as opposed to dict keys which
    come from :class:`IterableValidationNote` and are rendered as
    ``['key']``.
    """

    name: str


@frozen
class ErrorLeaf:
    """A leaf error at a path in the error tree.

    Attributes:
        path: The tuple of path components from the root to this error.
            Components are :class:`FieldName` (for class fields), ``int``
            (for list indexes), or ``str`` (for dict keys).
        error_type: The type of the exception.
        message: A formatted description of the error.
        cause_types: The types in the ``__cause__`` chain, in order.
            Cause values are not output, to avoid leaking input values.
    """

    path: _Path
    error_type: type
    message: str
    cause_types: tuple[type, ...]


@frozen
class ErrorBranch:
    """A branch in the error tree, grouping children under a path component.

    Attributes:
        component: The path component: a :class:`FieldName`, an ``int``
            index, or a ``str`` key.
        children: The child branches and leaves, in traversal order.
    """

    component: _PathComponent
    children: tuple[Union["ErrorBranch", "ErrorLeaf"], ...]


@frozen
class ErrorTree:
    """An organized error tree.

    Attributes:
        children: The root-level branches and leaves.
        total: The total number of leaf errors in the original exception.
        shown: The number of leaf errors present in the tree.
        truncated: Whether the tree was truncated because ``total > limit``.
        limit: The maximum number of leaf errors kept, or ``None`` for no
            limit.
    """

    children: tuple[Union["ErrorBranch", "ErrorLeaf"], ...]
    total: int
    shown: int
    truncated: bool
    limit: Optional[int]


def _note_for(exc: BaseException, note_type: type) -> Optional[Any]:
    """Find the first note of the given type on an exception, or None."""
    if not hasattr(exc, "__notes__"):
        return None
    for note in exc.__notes__:
        if note.__class__ is note_type:
            return note
    return None


def _collect_leaves(exc: BaseException, path: _Path, leaves: list[_RawLeaf]) -> None:
    """Recursively collect leaf errors with their paths.

    This is read-only and does not run structuring.
    """
    note_type: type
    if isinstance(exc, IterableValidationError):
        note_type = IterableValidationNote
    elif isinstance(exc, ClassValidationError):
        note_type = AttributeValidationNote
    else:
        # Not a validation error group - leaf at current path.
        leaves.append((path, exc, None))
        return

    for subexc in exc.exceptions:
        note = _note_for(subexc, note_type)
        if note is None:
            # No note - leaf at current path.
            leaves.append((path, subexc, None))
            continue

        if isinstance(note, IterableValidationNote):
            component: _PathComponent = note.index
        else:
            component = FieldName(note.name)
        subpath = (*path, component)
        expected_type = note.type

        if isinstance(subexc, BaseValidationError):
            # Nested validation error - recurse.
            _collect_leaves(subexc, subpath, leaves)
        else:
            # Leaf error with a path.
            leaves.append((subpath, subexc, expected_type))


def _collect_cause_types(exc: BaseException) -> tuple[type, ...]:
    """Collect the types in the cause chain, stopping on cycles.

    Only types are recorded, not values, to avoid leaking input values.
    """
    types: list[type] = []
    seen: set[int] = set()
    cause = getattr(exc, "__cause__", None)
    while cause is not None:
        cid = id(cause)
        if cid in seen:
            # Cycle detected - stop explicitly.
            break
        seen.add(cid)
        types.append(type(cause))
        cause = getattr(cause, "__cause__", None)
    return tuple(types)


def _make_leaf(
    path: _Path, exc: BaseException, expected_type: Any, formatter: Callable[..., str]
) -> ErrorLeaf:
    """Create an ErrorLeaf from an exception."""
    return ErrorLeaf(
        path=path,
        error_type=type(exc),
        message=formatter(exc, expected_type),
        cause_types=_collect_cause_types(exc),
    )


def _build_tree(
    leaves: list[tuple[_Path, ErrorLeaf]],
) -> tuple[Union["ErrorBranch", "ErrorLeaf"], ...]:
    """Build a tree from a list of ``(path, leaf)`` tuples.

    Returns a tuple of :class:`ErrorBranch` and :class:`ErrorLeaf`.
    """
    result: list[Union[ErrorBranch, ErrorLeaf]] = []
    groups: dict[_PathComponent, list[tuple[_Path, ErrorLeaf]]] = {}
    order: list[_PathComponent] = []

    for path, leaf in leaves:
        if not path:
            result.append(leaf)
            continue
        component = path[0]
        if component not in groups:
            groups[component] = []
            order.append(component)
        groups[component].append((path[1:], leaf))

    for component in order:
        children = _build_tree(groups[component])
        result.append(ErrorBranch(component=component, children=children))

    return tuple(result)


def build_error_tree(
    exc: Union[ClassValidationError, IterableValidationError, BaseException],
    limit: Optional[int] = None,
    formatter: Callable[..., str] = format_exception,
) -> ErrorTree:
    """Build a path tree from a validation error.

    This is read-only and does not run structuring. The original exception
    (and its notes) is left untouched.

    :param exc: The exception to organize. Should be an
        :class:`IterableValidationError` or :class:`ClassValidationError`.
    :param limit: The maximum number of leaf errors to keep. If ``None``, all
        leaves are kept. The total count is always recorded regardless of
        the limit.
    :param formatter: A callable to format exceptions into messages.
    :return: An :class:`ErrorTree`.
    """
    raw_leaves: list[_RawLeaf] = []
    _collect_leaves(exc, (), raw_leaves)

    total = len(raw_leaves)
    truncated = limit is not None and total > limit
    kept = raw_leaves[:limit] if truncated else raw_leaves

    leaves: list[tuple[_Path, ErrorLeaf]] = [
        (path, _make_leaf(path, subexc, expected_type, formatter))
        for path, subexc, expected_type in kept
    ]

    children = _build_tree(leaves)

    return ErrorTree(
        children=children,
        total=total,
        shown=len(leaves),
        truncated=truncated,
        limit=limit,
    )


def _render_component(component: Any) -> str:
    """Render a path component as a string."""
    if isinstance(component, FieldName):
        return f".{component.name}"
    return f"[{component!r}]"


def _render_leaf(leaf: ErrorLeaf) -> str:
    """Render a leaf error as a string."""
    parts = [f"✗ {leaf.error_type.__name__}: {leaf.message}"]
    if leaf.cause_types:
        chain = " → ".join(t.__name__ for t in leaf.cause_types)
        parts.append(f" (caused by: {chain})")
    return "".join(parts)


def _render_children(
    children: tuple[Union["ErrorBranch", "ErrorLeaf"], ...],
    indent: str,
    lines: list[str],
    truncated: bool,
    total: int,
    shown: int,
) -> None:
    """Render children with indentation."""
    for i, child in enumerate(children):
        is_last = (i == len(children) - 1) and not truncated
        connector = "└── " if is_last else "├── "
        if isinstance(child, ErrorLeaf):
            lines.append(f"{indent}{connector}{_render_leaf(child)}")
        else:
            comp_str = _render_component(child.component)
            # If this branch has a single leaf child, render on one line.
            if len(child.children) == 1 and isinstance(child.children[0], ErrorLeaf):
                leaf = child.children[0]
                lines.append(f"{indent}{connector}{comp_str} {_render_leaf(leaf)}")
            else:
                lines.append(f"{indent}{connector}{comp_str}")
                child_indent = indent + ("    " if is_last else "│   ")
                _render_children(
                    child.children, child_indent, lines, False, total, shown
                )
    if truncated:
        remaining = total - shown
        lines.append(
            f"{indent}└── ⋯ [truncated: {remaining} more error(s), total {total}]"
        )


def render_tree(tree: ErrorTree) -> str:
    """Render an error tree as a human-readable string.

    The tree is rendered with ``$`` as the root, branches for path
    components, and leaves for errors. A truncation marker is appended when
    the tree was truncated.
    """
    lines = ["$"]
    _render_children(tree.children, "", lines, tree.truncated, tree.total, tree.shown)
    return "\n".join(lines)
