"""Tests for the explicit hook selection explainer.

The explainer reports which hook (or hook factory) the real dispatch rules
of a converter would select for a type, without calling any hook or factory.

Run just these acceptance tests with:

    pytest tests/test_explain_dispatch.py -v
"""

from typing import Any

import pytest
from attrs import define

from cattrs import (
    BaseConverter,
    Converter,
    HookSelection,
    HookSelectionKind,
    PredicateRefusedError,
)
from cattrs.dispatch import MultiStrategyDispatch


@define
class A:
    x: int


@define
class B:
    y: str


class Counter:
    """A call-counting wrapper."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.fn(*args)


@pytest.fixture(params=(BaseConverter, Converter), scope="function")
def any_converter(request):
    return request.param()


def test_concrete_hook_beats_predicate(any_converter):
    """Concrete type hooks take priority over predicate hooks."""
    concrete = Counter(lambda v, t: A(v["x"]))
    pred = Counter(lambda t: True)
    pred_hook = Counter(lambda v, t: A(0))

    # The predicate is registered first, and would match anything.
    any_converter.register_structure_hook_func(pred, pred_hook)
    any_converter.register_structure_hook(A, concrete)

    sel = any_converter.explain_structure_hook(A)

    assert sel.kind is HookSelectionKind.SINGLE_DISPATCH
    assert sel.hook is concrete
    assert sel.matched_class is A
    # Nothing was called: no hooks, and not even the predicate.
    assert concrete.calls == 0
    assert pred_hook.calls == 0
    assert pred.calls == 0

    # The report matches the real dispatch.
    assert any_converter._structure_func.dispatch_without_caching(A) is concrete
    # And the real structure path uses the concrete hook.
    assert any_converter.structure({"x": 1}, A) == A(1)
    assert concrete.calls == 1
    assert pred_hook.calls == 0


def test_unstructure_concrete_hook_beats_predicate(any_converter):
    concrete = Counter(lambda o: {"x": o.x})
    pred_hook = Counter(lambda o: {})

    any_converter.register_unstructure_hook_func(lambda t: True, pred_hook)
    any_converter.register_unstructure_hook(A, concrete)

    sel = any_converter.explain_unstructure_hook(A)

    assert sel.kind is HookSelectionKind.SINGLE_DISPATCH
    assert sel.hook is concrete
    assert concrete.calls == 0
    assert pred_hook.calls == 0
    assert any_converter._unstructure_func.dispatch_without_caching(A) is concrete


def test_predicates_evaluated_once_at_real_position(any_converter):
    """Predicates run at most once, in real dispatch order, short-circuiting."""
    first = Counter(lambda t: False)
    second = Counter(lambda t: True)
    third = Counter(lambda t: True)  # never reached
    hook = Counter(lambda v, t: v)

    any_converter.register_structure_hook_func(third, hook)
    any_converter.register_structure_hook_func(second, hook)
    any_converter.register_structure_hook_func(first, hook)

    sel = any_converter.explain_structure_hook(B)

    assert sel.kind is HookSelectionKind.PREDICATE
    assert sel.hook is hook
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0  # evaluation stopped at the first match
    assert hook.calls == 0  # conversion functions are never called

    # Only the predicates up to and including the match are reported.
    assert [e.predicate for e in sel.predicate_evaluations] == [first, second]
    assert [e.result for e in sel.predicate_evaluations] == [False, True]
    assert all(not e.is_factory for e in sel.predicate_evaluations)

    # The report matches the real dispatch.
    assert any_converter._structure_func.dispatch_without_caching(B) is hook


def test_raising_predicate_is_skipped_like_real_dispatch(any_converter):
    def boom(t):
        raise TypeError("nope")

    good = Counter(lambda t: True)
    hook = Counter(lambda v, t: v)

    any_converter.register_structure_hook_func(good, hook)
    any_converter.register_structure_hook_func(boom, hook)

    sel = any_converter.explain_structure_hook(B)

    assert sel.kind is HookSelectionKind.PREDICATE
    assert sel.hook is hook
    skipped, matched = sel.predicate_evaluations
    assert skipped.result is None and isinstance(skipped.error, TypeError)
    assert matched.result is True
    assert any_converter._structure_func.dispatch_without_caching(B) is hook


def test_factory_selection_is_distinguished(any_converter):
    """Factory selections are reported separately, and factories not called."""
    fixed_hook = Counter(lambda v, t: v)
    factory = Counter(lambda t: fixed_hook)
    pred = Counter(lambda t: t is B)

    any_converter.register_structure_hook_factory(pred, factory)

    sel = any_converter.explain_structure_hook(B)

    assert sel.kind is HookSelectionKind.PREDICATE_FACTORY
    assert sel.is_factory
    assert sel.hook is factory  # the factory itself, uncalled
    assert factory.calls == 0
    assert fixed_hook.calls == 0
    assert pred.calls == 1
    assert sel.predicate_evaluations[-1].is_factory
    assert not sel.predicate_evaluations[-1].takes_converter

    # Realizing the selection matches the real dispatch.
    assert sel.realize() is fixed_hook
    assert any_converter._structure_func.dispatch_without_caching(B) is fixed_hook
    assert factory.calls == 2  # one call from realize, one from real dispatch


def test_extended_factory_selection_is_distinguished(any_converter):
    fixed_hook = Counter(lambda v, t: v)
    seen_converters = []

    def factory(t, converter):
        seen_converters.append(converter)
        return fixed_hook

    pred = Counter(lambda t: t is B)
    any_converter.register_structure_hook_factory(pred, factory)

    sel = any_converter.explain_structure_hook(B)

    assert sel.kind is HookSelectionKind.PREDICATE_FACTORY
    assert sel.hook is factory
    assert seen_converters == []
    assert sel.predicate_evaluations[-1].takes_converter

    with pytest.raises(ValueError, match="requires a converter"):
        sel.realize()

    assert sel.realize(any_converter) is fixed_hook
    assert seen_converters == [any_converter]
    assert any_converter._structure_func.dispatch_without_caching(B) is fixed_hook


def test_fallback_factory_selection():
    fallback = Counter(lambda t: lambda v, tt: v)
    converter = BaseConverter(structure_fallback_factory=fallback)

    class Unregistered:
        pass

    sel = converter.explain_structure_hook(Unregistered)

    assert sel.kind is HookSelectionKind.FALLBACK_FACTORY
    assert sel.is_factory
    assert sel.hook is fallback
    assert fallback.calls == 0  # the fallback is not called while explaining

    # The real dispatch would call it.
    assert converter._structure_func.dispatch_without_caching(Unregistered) is not None
    assert fallback.calls == 1


def test_refuse_unknown_predicates(any_converter):
    """Unknown (possibly side-effecting) predicates can be refused."""
    pred = Counter(lambda t: True)
    hook = Counter(lambda v, t: v)
    any_converter.register_structure_hook_func(pred, hook)

    with pytest.raises(PredicateRefusedError) as exc_info:
        any_converter.explain_structure_hook(B, allow_unknown_predicates=False)

    assert exc_info.value.predicate is pred
    assert exc_info.value.type_ is B
    assert pred.calls == 0  # the predicate was refused, not executed
    assert hook.calls == 0

    # Marking the predicate as pure allows the explanation.
    sel = any_converter.explain_structure_hook(
        B, allow_unknown_predicates=False, known_pure_predicates=[pred]
    )
    assert sel.kind is HookSelectionKind.PREDICATE
    assert sel.hook is hook
    assert pred.calls == 1


def test_refusal_is_lazy(any_converter):
    """Predicates past the first match are never reached, so not refused."""
    matching = Counter(lambda t: True)
    unknown = Counter(lambda t: True)
    hook = Counter(lambda v, t: v)

    any_converter.register_structure_hook_func(unknown, hook)
    any_converter.register_structure_hook_func(matching, hook)

    sel = any_converter.explain_structure_hook(
        B, allow_unknown_predicates=False, known_pure_predicates=[matching]
    )
    assert sel.kind is HookSelectionKind.PREDICATE
    assert matching.calls == 1
    assert unknown.calls == 0


def test_explain_does_not_touch_dispatch_cache(any_converter):
    """Explaining must not populate or clear the dispatch caches."""
    dispatch = any_converter._structure_func.dispatch
    any_converter.structure(1, int)  # warm the cache
    before = dispatch.cache_info()
    direct_before = dict(any_converter._structure_func._direct_dispatch)

    any_converter.explain_structure_hook(A)
    any_converter.explain_structure_hook(list[int])
    any_converter.explain_structure_hook(B)

    assert dispatch.cache_info() == before
    assert any_converter._structure_func._direct_dispatch == direct_before

    # Subsequent real dispatch results are unaffected.
    assert any_converter.structure({"x": 5}, A) == A(5)


def test_generic_identity_preserved(any_converter):
    """Generic aliases are used as-is; their parameter identity is kept."""
    li = list[int]
    hook = Counter(lambda v, t: v)
    any_converter.register_structure_hook_func(lambda t: t is li, hook)

    sel = any_converter.explain_structure_hook(li)
    assert sel.kind is HookSelectionKind.PREDICATE
    assert sel.hook is hook
    assert sel.type is li

    # A structurally equal but distinct alias is a different key.
    other = list[int]
    assert other is not li
    sel_other = any_converter.explain_structure_hook(other)
    assert sel_other.hook is not hook
    assert sel_other.type is other
    # And it still matches the real dispatch for that alias.
    assert sel_other.realize(any_converter)(
        [1], other
    ) == any_converter._structure_func.dispatch_without_caching(other)([1], other)


def test_direct_dispatch_generic_keys():
    """Direct dispatch keeps generic aliases as exact keys."""
    dispatch = MultiStrategyDispatch(lambda t: lambda v, tt: v, BaseConverter())
    int_hook = lambda v, t: v  # noqa: E731
    str_hook = lambda v, t: str(v)  # noqa: E731
    dispatch._direct_dispatch[list[int]] = int_hook
    dispatch._direct_dispatch[list[str]] = str_hook

    sel_int = dispatch.explain(list[int])
    assert sel_int.kind is HookSelectionKind.DIRECT_DISPATCH
    assert sel_int.hook is int_hook

    sel_str = dispatch.explain(list[str])
    assert sel_str.kind is HookSelectionKind.DIRECT_DISPATCH
    assert sel_str.hook is str_hook

    # list[bytes] is not registered directly; it falls through, like real dispatch.
    assert dispatch.explain(list[bytes]).kind is not HookSelectionKind.DIRECT_DISPATCH


def test_explain_matches_real_dispatch(any_converter):
    """The reported selection agrees with the real dispatch for many types."""
    fixed_hook = Counter(lambda v, t: v)
    any_converter.register_structure_hook_factory(
        lambda t: t is B, lambda t: fixed_hook
    )
    concrete = Counter(lambda v, t: A(v["x"]))
    any_converter.register_structure_hook(A, concrete)

    for t in (int, str, A, B, Any, list[int], dict[str, int]):
        sel = any_converter.explain_structure_hook(t)
        real = any_converter._structure_func.dispatch_without_caching(t)
        realized = sel.realize(any_converter)
        if sel.is_factory and realized is not fixed_hook:
            # Factories generate fresh hooks; compare behavior instead.
            sample = {"x": 1} if t is A else None
            if t == list[int]:
                sample = [1, 2]
            elif t == dict[str, int]:
                sample = {"a": 1}
            assert realized(sample, t) == real(sample, t)
        else:
            assert realized is real


def test_describe_renders_trace(any_converter):
    any_converter.register_structure_hook(A, lambda v, t: A(v["x"]))

    text = any_converter.explain_structure_hook(A).describe()
    assert "singledispatch" in text
    assert "A" in text

    text = any_converter.explain_structure_hook(list[int]).describe()
    assert "factory" in text
    assert "list[int]" in text


def test_hook_selection_defaults():
    hook = lambda v, t: v  # noqa: E731
    sel = HookSelection(int, HookSelectionKind.PREDICATE, hook)
    assert not sel.is_factory
    assert sel.predicate_evaluations == []
    assert sel.matched_class is None
    assert sel.realize() is hook
