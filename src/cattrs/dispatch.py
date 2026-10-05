from __future__ import annotations

from enum import Enum
from functools import lru_cache, singledispatch
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, Literal, TypeVar

from attrs import Factory, define

from ._compat import TypeAlias
from .errors import PredicateRefusedError
from .fns import Predicate

if TYPE_CHECKING:
    from .converters import BaseConverter

TargetType: TypeAlias = Any
UnstructuredValue: TypeAlias = Any
StructuredValue: TypeAlias = Any

StructureHook: TypeAlias = Callable[[UnstructuredValue, TargetType], StructuredValue]
UnstructureHook: TypeAlias = Callable[[StructuredValue], UnstructuredValue]

Hook = TypeVar("Hook", StructureHook, UnstructureHook)
HookFactory: TypeAlias = Callable[[TargetType], Hook]


@define
class _DispatchNotFound:
    """A dummy object to help signify a dispatch not found."""


class HookSelectionKind(Enum):
    """The dispatch stage at which a hook (or hook factory) was selected.

    The stages mirror the real dispatch order: concrete class hooks
    (:meth:`functools.singledispatch`), exact-type direct dispatch, predicate
    hooks and predicate hook factories (newest registration first), and
    finally the fallback factory.
    """

    SINGLE_DISPATCH = "single_dispatch"
    DIRECT_DISPATCH = "direct_dispatch"
    PREDICATE = "predicate"
    PREDICATE_FACTORY = "predicate_factory"
    FALLBACK_FACTORY = "fallback_factory"


@define
class PredicateEvaluation:
    """The outcome of evaluating a single predicate while explaining dispatch.

    Each predicate is evaluated at most once, exactly at the position where
    the real dispatch would evaluate it.
    """

    predicate: Predicate
    is_factory: bool
    takes_converter: bool
    result: bool | None
    error: Exception | None = None


@define
class HookSelection(Generic[Hook]):
    """The result of explaining hook selection for a type.

    :ivar type: The type exactly as it was explained; generic parameters keep
        their identity (no normalization to origins is performed).
    :ivar kind: The dispatch stage which produced the selection.
    :ivar hook: The selected hook for `SINGLE_DISPATCH`, `DIRECT_DISPATCH` and
        `PREDICATE` selections. For `PREDICATE_FACTORY` and `FALLBACK_FACTORY`
        selections this is the (uncalled) hook factory itself, so explaining
        never runs factories or hooks.
    :ivar predicate_evaluations: The predicate evaluations performed, in
        order. Only predicates up to and including the first match are
        present.
    :ivar matched_class: For `SINGLE_DISPATCH` selections, the registered
        class which matched, if it could be determined.

    ..  versionadded:: NEXT
    """

    type: TargetType
    kind: HookSelectionKind
    hook: Any
    predicate_evaluations: list[PredicateEvaluation] = Factory(list)
    matched_class: type | None = None

    @property
    def is_factory(self) -> bool:
        """Whether the selection is a hook factory (rather than a hook)."""
        return self.kind in (
            HookSelectionKind.PREDICATE_FACTORY,
            HookSelectionKind.FALLBACK_FACTORY,
        )

    def realize(self, converter: BaseConverter | None = None) -> Hook:
        """Produce the hook the real dispatch would return for this selection.

        For factory selections this calls the selected factory (passing
        `converter` for extended factories), so unlike explaining, this is
        not side-effect free.
        """
        if self.kind is HookSelectionKind.FALLBACK_FACTORY:
            return self.hook(self.type)
        if self.kind is HookSelectionKind.PREDICATE_FACTORY:
            if self.predicate_evaluations[-1].takes_converter:
                if converter is None:
                    msg = (
                        "The selected factory requires a converter; "
                        "pass one via `converter`."
                    )
                    raise ValueError(msg)
                return self.hook(self.type, converter)
            return self.hook(self.type)
        return self.hook

    def describe(self) -> str:
        """Render a human-readable trace of the selection."""
        lines = [f"Hook selection for {self.type!r}:"]
        if self.kind is HookSelectionKind.SINGLE_DISPATCH:
            lines.append(
                f"  concrete class hook (singledispatch): hit {self.hook!r}"
                + (
                    f" (registered for {self.matched_class!r})"
                    if self.matched_class is not None
                    else ""
                )
            )
        elif self.kind is HookSelectionKind.DIRECT_DISPATCH:
            lines.append(f"  direct dispatch: hit {self.hook!r}")
        else:
            for i, ev in enumerate(self.predicate_evaluations, 1):
                role = "factory predicate" if ev.is_factory else "predicate"
                if ev.error is not None:
                    outcome = f"raised {ev.error!r} (skipped)"
                else:
                    outcome = str(ev.result)
                lines.append(f"  {i}. {role} {ev.predicate!r}: {outcome}")
            if self.kind is HookSelectionKind.PREDICATE:
                lines.append(f"  => predicate hook selected: {self.hook!r}")
            elif self.kind is HookSelectionKind.PREDICATE_FACTORY:
                lines.append(f"  => hook factory selected (not called): {self.hook!r}")
            else:
                lines.append(
                    f"  => fallback factory selected (not called): {self.hook!r}"
                )
        return "\n".join(lines)


@define
class FunctionDispatch:
    """
    FunctionDispatch is similar to functools.singledispatch, but
    instead dispatches based on functions that take the type of the
    first argument in the method, and return True or False.

    objects that help determine dispatch should be instantiated objects.

    :param converter: A converter to be used for factories that require converters.

    ..  versionchanged:: 24.1.0
        Support for factories that require converters, hence this requires a
        converter when creating.
    """

    _converter: BaseConverter
    _handler_pairs: list[tuple[Predicate, Callable[[Any, Any], Any], bool, bool]] = (
        Factory(list)
    )

    def register(
        self,
        predicate: Predicate,
        func: Callable[..., Any],
        is_generator=False,
        takes_converter=False,
    ) -> None:
        self._handler_pairs.insert(0, (predicate, func, is_generator, takes_converter))

    def dispatch(self, typ: Any) -> Callable[..., Any] | None:
        """
        Return the appropriate handler for the object passed.
        """
        for can_handle, handler, is_generator, takes_converter in self._handler_pairs:
            # can handle could raise an exception here
            # such as issubclass being called on an instance.
            # it's easier to just ignore that case.
            try:
                ch = can_handle(typ)
            except Exception:  # noqa: S112
                continue
            if ch:
                if is_generator:
                    if takes_converter:
                        return handler(typ, self._converter)
                    return handler(typ)

                return handler
        return None

    def get_num_fns(self) -> int:
        return len(self._handler_pairs)

    def explain(
        self,
        typ: Any,
        *,
        allow_unknown_predicates: bool = True,
        known_pure_predicates: Iterable[Predicate] = (),
    ) -> tuple[HookSelectionKind | None, Any, list[PredicateEvaluation]]:
        """Explain which handler would be selected for `typ`.

        Follows the real dispatch rules: predicates are evaluated in
        registration order (newest first), each at most once, stopping at the
        first match. Handlers and factories are never called.

        :param allow_unknown_predicates: If False, refuse (by raising
            :class:`cattrs.errors.PredicateRefusedError`) to evaluate any
            predicate not present in `known_pure_predicates`, since it may
            have side effects.
        :param known_pure_predicates: Predicates known to be side-effect free,
            matched by identity.
        """
        known = {id(p) for p in known_pure_predicates}
        evaluations: list[PredicateEvaluation] = []
        for can_handle, handler, is_generator, takes_converter in self._handler_pairs:
            if not allow_unknown_predicates and id(can_handle) not in known:
                raise PredicateRefusedError(can_handle, typ)
            try:
                ch = can_handle(typ)
            except Exception as exc:
                # Mirror the real dispatch: a raising predicate is skipped.
                evaluations.append(
                    PredicateEvaluation(
                        can_handle, bool(is_generator), takes_converter, None, exc
                    )
                )
                continue
            evaluations.append(
                PredicateEvaluation(
                    can_handle, bool(is_generator), takes_converter, bool(ch)
                )
            )
            if ch:
                kind = (
                    HookSelectionKind.PREDICATE_FACTORY
                    if is_generator
                    else HookSelectionKind.PREDICATE
                )
                return kind, handler, evaluations
        return None, None, evaluations

    def copy_to(self, other: FunctionDispatch, skip: int = 0) -> None:
        other._handler_pairs = self._handler_pairs[:-skip] + other._handler_pairs


@define(init=False)
class MultiStrategyDispatch(Generic[Hook]):
    """
    MultiStrategyDispatch uses a combination of exact-match dispatch,
    singledispatch, and FunctionDispatch.

    :param fallback_factory: A hook factory to be called when a hook cannot be
        produced.
    :param converter: A converter to be used for factories that require converters.

    .. versionchanged:: 23.2.0
        Fallbacks are now factories.
    .. versionchanged:: 24.1.0
        Support for factories that require converters, hence this requires a
        converter when creating.
    """

    _fallback_factory: HookFactory[Hook]
    _direct_dispatch: dict[TargetType, Hook]
    _function_dispatch: FunctionDispatch
    _single_dispatch: Any
    dispatch: Callable[[TargetType, BaseConverter], Hook]

    def __init__(
        self, fallback_factory: HookFactory[Hook], converter: BaseConverter
    ) -> None:
        self._fallback_factory = fallback_factory
        self._direct_dispatch = {}
        self._function_dispatch = FunctionDispatch(converter)
        self._single_dispatch = singledispatch(_DispatchNotFound)
        self.dispatch = lru_cache(maxsize=None)(self.dispatch_without_caching)

    def dispatch_without_caching(self, typ: TargetType) -> Hook:
        """Dispatch on the type but without caching the result."""
        try:
            dispatch = self._single_dispatch.dispatch(typ)
            if dispatch is not _DispatchNotFound:
                return dispatch
        except Exception:  # noqa: S110
            pass

        direct_dispatch = self._direct_dispatch.get(typ)
        if direct_dispatch is not None:
            return direct_dispatch

        res = self._function_dispatch.dispatch(typ)
        return res if res is not None else self._fallback_factory(typ)

    def explain(
        self,
        typ: TargetType,
        *,
        allow_unknown_predicates: bool = True,
        known_pure_predicates: Iterable[Predicate] = (),
    ) -> HookSelection[Hook]:
        """Explain which hook would be selected for `typ`.

        Follows the real dispatch rules (concrete class hooks first, then
        direct dispatch, then predicate hooks and hook factories in
        registration order, then the fallback factory) without calling any
        hook or hook factory, and without touching the dispatch caches, so
        explaining never influences subsequent real dispatch results.

        The type is used as-is, so generic parameters keep their identity.

        :param allow_unknown_predicates: If False, refuse (by raising
            :class:`cattrs.errors.PredicateRefusedError`) to evaluate any
            predicate not present in `known_pure_predicates`, since it may
            have side effects.
        :param known_pure_predicates: Predicates known to be side-effect free,
            matched by identity.

        ..  versionadded:: NEXT
        """
        try:
            dispatch = self._single_dispatch.dispatch(typ)
            if dispatch is not _DispatchNotFound:
                matched_class = next(
                    (
                        c
                        for c, f in self._single_dispatch.registry.items()
                        if f is dispatch
                    ),
                    None,
                )
                return HookSelection(
                    typ,
                    HookSelectionKind.SINGLE_DISPATCH,
                    dispatch,
                    matched_class=matched_class,
                )
        except Exception:  # noqa: S110
            pass

        direct_dispatch = self._direct_dispatch.get(typ)
        if direct_dispatch is not None:
            return HookSelection(
                typ, HookSelectionKind.DIRECT_DISPATCH, direct_dispatch
            )

        kind, handler, evaluations = self._function_dispatch.explain(
            typ,
            allow_unknown_predicates=allow_unknown_predicates,
            known_pure_predicates=known_pure_predicates,
        )
        if kind is not None:
            return HookSelection(typ, kind, handler, evaluations)
        return HookSelection(
            typ, HookSelectionKind.FALLBACK_FACTORY, self._fallback_factory, evaluations
        )

    def register_cls_list(self, cls_and_handler, direct: bool = False) -> None:
        """Register a class to direct or singledispatch."""
        for cls, handler in cls_and_handler:
            if direct:
                self._direct_dispatch[cls] = handler
            else:
                self._single_dispatch.register(cls, handler)
                self.clear_direct()
        self.dispatch.cache_clear()

    def register_func_list(
        self,
        pred_and_handler: list[
            tuple[Predicate, Any]
            | tuple[Predicate, Any, bool]
            | tuple[Predicate, Callable[[Any, BaseConverter], Any], Literal["extended"]]
        ],
    ):
        """
        Register a predicate function to determine if the handler
        should be used for the type.

        :param pred_and_handler: The list of predicates and their associated
            handlers. If a handler is registered in `extended` mode, it's a
            factory that requires a converter.
        """
        for tup in pred_and_handler:
            if len(tup) == 2:
                func, handler = tup
                self._function_dispatch.register(func, handler)
            else:
                func, handler, is_gen = tup
                if is_gen == "extended":
                    self._function_dispatch.register(
                        func, handler, is_generator=is_gen, takes_converter=True
                    )
                else:
                    self._function_dispatch.register(func, handler, is_generator=is_gen)
        self.clear_direct()
        self.dispatch.cache_clear()

    def clear_direct(self) -> None:
        """Clear the direct dispatch."""
        self._direct_dispatch.clear()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._direct_dispatch.clear()
        self.dispatch.cache_clear()

    def get_num_fns(self) -> int:
        return self._function_dispatch.get_num_fns()

    def copy_to(self, other: MultiStrategyDispatch, skip: int = 0) -> None:
        self._function_dispatch.copy_to(other._function_dispatch, skip=skip)
        for cls, fn in self._single_dispatch.registry.items():
            other._single_dispatch.register(cls, fn)
        other.clear_cache()
