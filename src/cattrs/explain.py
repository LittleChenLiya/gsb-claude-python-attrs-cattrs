"""Explicit hook-selection explanation.

This module explains *which* hook a converter would select for a given type,
following the real dispatch rules of
:class:`cattrs.dispatch.MultiStrategyDispatch`, without ever calling a
conversion function.

The real dispatch order is:

1. class hooks registered through singledispatch
   (:meth:`~cattrs.BaseConverter.register_structure_hook` /
   :meth:`~cattrs.BaseConverter.register_unstructure_hook` for regular
   classes),
2. direct hooks (generated hooks cached for specific generic types),
3. predicate hooks and hook factories
   (:meth:`~cattrs.BaseConverter.register_structure_hook_func` /
   :meth:`~cattrs.BaseConverter.register_structure_hook_factory`), in reverse
   registration order,
4. the fallback factory.

The explanation:

* never calls the selected hook (the actual conversion function),
* evaluates every predicate at most once, at the exact position the real
  dispatch would evaluate it,
* preserves the exact type object passed in (generic parameter identity),
* optionally refuses to evaluate unknown predicates, which may have side
  effects,
* never modifies the converter's dispatch cache.

Priority example::

    >>> from cattrs import BaseConverter
    >>> from cattrs.explain import explain_structure_hook
    >>> conv = BaseConverter()
    >>> class Foo: ...
    >>> conv.register_structure_hook(Foo, lambda v, _: ("hook", v))
    >>> conv.register_structure_hook_func(
    ...     lambda t: t is Foo, lambda v, _: ("pred", v)
    ... )
    >>> expl = explain_structure_hook(conv, Foo)
    >>> expl.kind
    'singledispatch'
    >>> expl.hook(1, Foo)
    ('hook', 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from attrs import Factory, define

from .dispatch import _DispatchNotFound
from .errors import CattrsError

if TYPE_CHECKING:
    from .converters import BaseConverter
    from .dispatch import MultiStrategyDispatch
    from .fns import Predicate

Direction = Literal["structure", "unstructure"]

#: The kind of selection that was made.
#:
#: * ``singledispatch``: a class hook registered through singledispatch,
#: * ``direct``: a generated hook cached for a specific generic type,
#: * ``predicate``: a hook matched by a predicate,
#: * ``factory``: a hook factory matched by a predicate (not called),
#: * ``fallback``: the fallback factory (not called).
HookKind = Literal["singledispatch", "direct", "predicate", "factory", "fallback"]


class UnknownPredicateError(CattrsError):
    """Raised when an explanation encounters a predicate it cannot trust.

    Unknown predicates (predicates not shipped with cattrs) may have side
    effects, so the explanation refuses to call them when
    ``reject_unknown_predicates`` is enabled.
    """

    def __init__(self, predicate: Predicate, typ: Any) -> None:
        self.predicate = predicate
        self.typ = typ
        super().__init__(
            f"Refusing to evaluate unknown predicate {predicate!r} for type "
            f"{typ!r}; it may have side effects. Call the explanation with "
            "reject_unknown_predicates=False to evaluate it anyway."
        )


@define
class PredicateEvaluation:
    """The result of evaluating a single predicate during explanation.

    :param predicate: The predicate that was evaluated.
    :param result: What the predicate returned (``False`` when it raised).
    :param index: The position of the predicate in the function dispatch
        handler pairs.
    :param raised: Whether the predicate raised instead of returning.
    """

    predicate: Predicate
    result: bool
    index: int
    raised: bool = False


@define
class HookExplanation:
    """The result of explaining hook selection for a type.

    :param typ: The type the explanation was run for. This is the exact
        object passed in, so generic parameter identity is preserved.
    :param direction: Whether this explains structure or unstructure hooks.
    :param kind: Which stage selected the hook.
    :param hook: The selected hook, for ``singledispatch``, ``direct`` and
        ``predicate`` selections.
    :param factory: The selected factory, for ``factory`` and ``fallback``
        selections. The factory is not called by the explanation.
    :param predicate: The predicate that matched, for ``predicate`` and
        ``factory`` selections.
    :param is_generator: Whether the matched predicate was registered as a
        factory.
    :param takes_converter: Whether the factory also takes the converter.
    :param evaluations: The predicates evaluated while explaining, in order.
    """

    typ: Any
    direction: Direction
    kind: HookKind
    hook: Any = None
    factory: Any = None
    predicate: Predicate | None = None
    is_generator: bool = False
    takes_converter: bool = False
    evaluations: list[PredicateEvaluation] = Factory(list)

    @property
    def selected(self) -> Any:
        """The selected callable.

        This is the hook for ``singledispatch``, ``direct`` and ``predicate``
        selections, and the factory for ``factory`` and ``fallback``
        selections. Neither is called by the explanation.
        """
        return self.hook if self.hook is not None else self.factory


def _is_known_predicate(predicate: Any) -> bool:
    """Is *predicate* shipped with cattrs (and thus safe to call)?

    User-provided predicates are unknown and may have side effects. Wrappers
    such as :class:`functools.partial` are unwrapped.
    """
    module = getattr(predicate, "__module__", None)
    if isinstance(module, str) and (module == "cattrs" or module.startswith("cattrs.")):
        return True
    wrapped = getattr(predicate, "func", None)
    if wrapped is not None and wrapped is not predicate:
        return _is_known_predicate(wrapped)
    return False


def _explain_dispatch(
    dispatch: MultiStrategyDispatch,
    typ: Any,
    direction: Direction,
    *,
    reject_unknown_predicates: bool,
) -> HookExplanation:
    # Stage 1: class hooks registered through singledispatch.
    # singledispatch.dispatch may raise for types it cannot handle (such as
    # non-types), mirror the real dispatch and ignore that case.
    try:
        handler = dispatch._single_dispatch.dispatch(typ)
        if handler is not _DispatchNotFound:
            return HookExplanation(typ, direction, "singledispatch", hook=handler)
    except Exception:  # noqa: S110
        pass

    # Stage 2: direct dispatch (generated hooks for specific generic types).
    # This is a plain dict lookup, mirror the real dispatch (which lets
    # errors propagate).
    handler = dispatch._direct_dispatch.get(typ)
    if handler is not None:
        return HookExplanation(typ, direction, "direct", hook=handler)

    # Stage 3: predicate hooks and hook factories.
    # Mirror FunctionDispatch.dispatch, but never call the handlers or
    # factories. Each predicate is evaluated at most once, in registration
    # order, and we stop at the first match.
    evaluations: list[PredicateEvaluation] = []
    for index, (predicate, handler, is_generator, takes_converter) in enumerate(
        dispatch._function_dispatch._handler_pairs
    ):
        if reject_unknown_predicates and not _is_known_predicate(predicate):
            raise UnknownPredicateError(predicate, typ)
        try:
            result = bool(predicate(typ))
        except Exception:
            # The real dispatch ignores predicates that raise and keeps
            # looking.
            evaluations.append(
                PredicateEvaluation(predicate, False, index, raised=True)
            )
            continue
        evaluations.append(PredicateEvaluation(predicate, result, index))
        if result:
            if is_generator:
                return HookExplanation(
                    typ,
                    direction,
                    "factory",
                    factory=handler,
                    predicate=predicate,
                    is_generator=True,
                    takes_converter=takes_converter,
                    evaluations=evaluations,
                )
            return HookExplanation(
                typ,
                direction,
                "predicate",
                hook=handler,
                predicate=predicate,
                evaluations=evaluations,
            )

    # Stage 4: fallback factory. The real dispatch would call it; we don't.
    return HookExplanation(
        typ,
        direction,
        "fallback",
        factory=dispatch._fallback_factory,
        evaluations=evaluations,
    )


def explain_structure_hook(
    converter: BaseConverter, typ: Any, *, reject_unknown_predicates: bool = False
) -> HookExplanation:
    """Explain which structure hook *converter* would select for *typ*.

    The explanation follows the real dispatch rules of the converter without
    calling any conversion function.

    :param converter: The converter to explain.
    :param typ: The type to explain the selection for. The exact object is
        preserved, so generic parameter identity is kept.
    :param reject_unknown_predicates: If ``True``, refuse to evaluate
        predicates not shipped with cattrs, since they may have side
        effects, and raise :class:`UnknownPredicateError` instead.

    .. versionadded:: NEXT
    """
    return _explain_dispatch(
        converter._structure_func,
        typ,
        "structure",
        reject_unknown_predicates=reject_unknown_predicates,
    )


def explain_unstructure_hook(
    converter: BaseConverter, typ: Any, *, reject_unknown_predicates: bool = False
) -> HookExplanation:
    """Explain which unstructure hook *converter* would select for *typ*.

    The explanation follows the real dispatch rules of the converter without
    calling any conversion function.

    :param converter: The converter to explain.
    :param typ: The type to explain the selection for. The exact object is
        preserved, so generic parameter identity is kept.
    :param reject_unknown_predicates: If ``True``, refuse to evaluate
        predicates not shipped with cattrs, since they may have side
        effects, and raise :class:`UnknownPredicateError` instead.

    .. versionadded:: NEXT
    """
    return _explain_dispatch(
        converter._unstructure_func,
        typ,
        "unstructure",
        reject_unknown_predicates=reject_unknown_predicates,
    )
