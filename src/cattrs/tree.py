"""An explicit, read-only layer organizing validation errors into a path tree.

This layer turns :class:`cattrs.IterableValidationError` and
:class:`cattrs.ClassValidationError` into an immutable tree of nodes keyed by
the *real* attribute names, list indices and mapping keys where the failures
happened.

Unlike :func:`cattrs.transform_error` (which produces a flat list of strings),
:func:`build_error_tree` produces structured data and supports capping the
number of kept leaf errors via ``max_leaves`` while still preserving the
*total* number of failures. Truncation is marked independently
(:attr:`ErrorTree.truncated`, :attr:`Node.dropped`), so a capped tree can never
be mistaken for a fully-passing or fully-enumerated one.

The layer is strictly read-only: it only inspects the exception, its notes and
its ``__cause__`` chain. It never re-runs structuring, never mutates the
exception and never records input values -- leaf causes are recorded as
exception *type names* only. Cyclic ``__cause__`` chains (and cyclic exception
groups) are detected and stopped explicitly with markers instead of recursing
forever.

Example
-------

>>> from cattrs import structure
>>> from cattrs.errors import IterableValidationError
>>> from cattrs.tree import build_error_tree

>>> try:
...     structure([["a"], ["b"], ["c"], ["d"], ["e"]], list[list[int]])
... except IterableValidationError as exc:
...     tree = build_error_tree(exc, max_leaves=2)
>>> tree.total
5
>>> tree.truncated
True
>>> sorted(leaf.path_str for leaf in tree.leaves())
['$[0][0]', '$[1][0]']

Run the tests for this module with::

    python -m pytest tests/test_error_tree.py -v

.. versionadded:: 25.1.0
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal, Optional, Union

from attrs import field, frozen

from .errors import (
    ClassValidationError,
    ForbiddenExtraKeysError,
    IterableValidationError,
)

__all__ = [
    "CycleNode",
    "ErrorTree",
    "Leaf",
    "Node",
    "Segment",
    "build_error_tree",
    "format_path",
    "safe_format_exception",
]

#: Groups which are recursively decomposed into child segments.
_GROUP_TYPES = (IterableValidationError, ClassValidationError)


@frozen
class Segment:
    """A single, real step in an error path.

    Segments are never renumbered or reordered, even when sibling errors are
    dropped due to the leaf cap.
    """

    kind: Literal["attribute", "index", "key"]
    value: Union[str, int]

    def __str__(self) -> str:
        if self.kind == "attribute":
            return f".{self.value}"
        if self.kind == "index":
            return f"[{self.value}]"
        return f"[{self.value!r}]"


def format_path(segments: tuple[Segment, ...], root: str = "$") -> str:
    """Render a tuple of segments into a path string like ``$.a[0]['k']``."""
    return root + "".join(str(s) for s in segments)


def safe_format_exception(exc: BaseException, type: Union[type, None]) -> str:
    """Like :func:`cattrs.v.format_exception`, but never interpolates the
    exception instance itself, so input values are never leaked into messages.
    """
    if isinstance(exc, KeyError):
        res = "required field missing"
    elif isinstance(exc, ValueError):
        if type is not None:
            tn = type.__name__ if hasattr(type, "__name__") else repr(type)
            res = f"invalid value for type, expected {tn}"
        else:
            res = "invalid value"
    elif isinstance(exc, TypeError):
        if type is None:
            if exc.args and str(exc.args[0]).endswith("object is not iterable"):
                res = "invalid value for type, expected an iterable"
            else:
                res = f"invalid type ({exc.__class__.__name__})"
        else:
            tn = type.__name__ if hasattr(type, "__name__") else repr(type)
            res = f"invalid value for type, expected {tn}"
    elif isinstance(exc, ForbiddenExtraKeysError):
        res = f"extra fields found ({', '.join(sorted(exc.extra_fields))})"
    elif (
        isinstance(exc, AttributeError)
        and exc.args
        and str(exc.args[0]).endswith("object has no attribute 'items'")
    ):
        # This was supposed to be a mapping (have .items()) but is not.
        res = "expected a mapping"
    else:
        res = f"unknown error ({exc.__class__.__name__})"

    return res


def _cause_chain(exc: BaseException) -> tuple[tuple[str, ...], bool]:
    """Collect the type names of ``exc`` and its ``__cause__`` chain.

    Only type names are kept, never ``str(exc)`` (which may contain input
    values). A cyclic chain is stopped explicitly; the second return value
    flags whether a cycle was detected.
    """
    types = [exc.__class__.__name__]
    seen = {id(exc)}
    cycle = False
    cause = exc.__cause__
    while cause is not None:
        if id(cause) in seen:
            # A cyclic cause chain: stop explicitly instead of looping.
            cycle = True
            break
        seen.add(id(cause))
        types.append(cause.__class__.__name__)
        cause = cause.__cause__
    return tuple(types), cycle


@frozen
class Leaf:
    """A single kept validation failure, at a real path."""

    path: tuple[Segment, ...]
    message: str
    #: Type names of the failing exception and its ``__cause__`` chain.
    cause_types: tuple[str, ...]
    #: Whether the ``__cause__`` chain was cyclic and had to be stopped.
    cause_cycle: bool = False

    @property
    def path_str(self) -> str:
        """This leaf's path, rendered like ``$.a[0]``."""
        return format_path(self.path)


@frozen
class CycleNode:
    """An explicit marker where a cyclic exception group chain was stopped."""

    exc_type: str


@frozen
class Node:
    """A branch of the error tree.

    Children are keyed by real segments, in original encounter order.
    """

    children: tuple[tuple[Segment, Union[Node, CycleNode]], ...] = ()
    #: Failures exactly at this node's own path.
    errors: tuple[Leaf, ...] = ()
    #: Number of failures at or below this node omitted due to the leaf cap.
    dropped: int = 0

    def iter_leaves(self, path: tuple[Segment, ...] = ()) -> Iterator[Leaf]:
        """Iterate over all kept leaves at or below this node."""
        yield from self.errors
        for segment, child in self.children:
            if isinstance(child, Node):
                yield from child.iter_leaves((*path, segment))


@frozen
class ErrorTree:
    """The result of organizing a validation error into a tree.

    The tree is immutable and independent of the source exception.
    """

    root: Node
    #: The total number of leaf failures found, kept or dropped.
    total: int
    #: The cap on kept leaves this tree was built with, if any.
    max_leaves: Optional[int] = field(default=None)

    @property
    def kept(self) -> int:
        """The number of leaf failures actually kept in the tree."""
        return sum(1 for _ in self.root.iter_leaves())

    @property
    def dropped(self) -> int:
        """The number of leaf failures omitted due to ``max_leaves``."""
        return self.root.dropped

    @property
    def truncated(self) -> bool:
        """Whether failures were omitted due to ``max_leaves``.

        This is an independent marker: a truncated tree never looks like a
        complete enumeration of the failures.
        """
        return self.dropped > 0

    @property
    def is_complete(self) -> bool:
        """Whether every failure is represented in the tree."""
        return not self.truncated

    @property
    def ok(self) -> bool:
        """Whether there were no failures at all (and none were hidden)."""
        return self.total == 0 and not self.truncated

    def leaves(self) -> Iterator[Leaf]:
        """Iterate over all kept leaves, in encounter order."""
        return self.root.iter_leaves()


class _Builder:
    """A mutable node under construction."""

    __slots__ = ("children", "dropped", "errors")

    def __init__(self) -> None:
        self.children: dict[Segment, Union[_Builder, CycleNode]] = {}
        self.errors: list[Leaf] = []
        self.dropped = 0


def _freeze(builder: _Builder) -> Node:
    children: list[tuple[Segment, Union[Node, CycleNode]]] = []
    dropped = builder.dropped
    for segment, child in builder.children.items():
        if isinstance(child, CycleNode):
            children.append((segment, child))
        else:
            frozen_child = _freeze(child)
            dropped += frozen_child.dropped
            children.append((segment, frozen_child))
    return Node(tuple(children), tuple(builder.errors), dropped)


class _State:
    __slots__ = ("format", "group_stack", "kept", "max_leaves", "total")

    def __init__(
        self,
        max_leaves: Optional[int],
        format_exception: Callable[[BaseException, Union[type, None]], str],
    ) -> None:
        self.max_leaves = max_leaves
        self.format = format_exception
        self.total = 0
        self.kept = 0
        #: ids of group exceptions on the current path, for cycle detection.
        self.group_stack: set[int] = set()


def _record_leaf(
    exc: BaseException,
    note_type: Union[type, None],
    builder: _Builder,
    state: _State,
    path: tuple[Segment, ...],
) -> None:
    state.total += 1
    if state.max_leaves is None or state.kept < state.max_leaves:
        state.kept += 1
        cause_types, cause_cycle = _cause_chain(exc)
        builder.errors.append(
            Leaf(path, state.format(exc, note_type), cause_types, cause_cycle)
        )
    else:
        # Over the cap: count it, but keep nothing and renumber nothing.
        builder.dropped += 1


def _descend(
    subexc: BaseException,
    note_type: Union[type, None],
    segment: Segment,
    builder: _Builder,
    state: _State,
    path: tuple[Segment, ...],
) -> None:
    child_path = (*path, segment)
    if isinstance(subexc, _GROUP_TYPES):
        if id(subexc) in state.group_stack:
            # A cyclic exception group chain: stop explicitly with a marker.
            builder.children[segment] = CycleNode(subexc.__class__.__name__)
            return
        child = _Builder()
        builder.children[segment] = child
        _walk_group(subexc, child, state, child_path)
    else:
        child = builder.children.get(segment)
        if child is None or isinstance(child, CycleNode):
            child = _Builder()
            builder.children[segment] = child
        _record_leaf(subexc, note_type, child, state, child_path)


def _walk_group(
    exc: Union[IterableValidationError, ClassValidationError],
    builder: _Builder,
    state: _State,
    path: tuple[Segment, ...],
) -> None:
    state.group_stack.add(id(exc))
    try:
        with_notes, without = exc.group_exceptions()
        if isinstance(exc, IterableValidationError):
            for subexc, note in with_notes:
                if isinstance(note.index, int):
                    segment = Segment("index", note.index)
                else:
                    segment = Segment("key", note.index)
                _descend(subexc, note.type, segment, builder, state, path)
        else:
            for subexc, note in with_notes:
                segment = Segment("attribute", note.name)
                _descend(subexc, note.type, segment, builder, state, path)
        for subexc in without:
            _record_leaf(subexc, None, builder, state, path)
    finally:
        state.group_stack.discard(id(exc))


def build_error_tree(
    exc: Union[ClassValidationError, IterableValidationError, BaseException],
    *,
    max_leaves: Optional[int] = None,
    format_exception: Callable[
        [BaseException, Union[type, None]], str
    ] = safe_format_exception,
) -> ErrorTree:
    """Organize a validation error into an immutable path tree.

    This is a read-only operation: the exception (including its notes) is only
    inspected, never modified, and structuring is never re-run.

    :param exc: The exception to organize, usually produced by a converter
        with ``detailed_validation`` enabled.
    :param max_leaves: The maximum number of leaf failures to keep. All
        failures are still counted into :attr:`ErrorTree.total`, and omitted
        ones are marked via :attr:`ErrorTree.truncated` and
        :attr:`Node.dropped`. ``None`` (the default) keeps everything.
    :param format_exception: A callable turning leaf exceptions into messages.
        The default, :func:`safe_format_exception`, never includes input
        values.

    .. versionadded:: 25.1.0
    """
    if max_leaves is not None and max_leaves < 0:
        raise ValueError("max_leaves must be non-negative or None")

    state = _State(max_leaves, format_exception)
    root = _Builder()
    if isinstance(exc, _GROUP_TYPES):
        _walk_group(exc, root, state, ())
    else:
        _record_leaf(exc, None, root, state, ())

    return ErrorTree(_freeze(root), state.total, max_leaves)
