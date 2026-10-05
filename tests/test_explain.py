"""Tests for the explicit hook-selection explainer."""

from __future__ import annotations

import pytest

from cattrs import BaseConverter, Converter
from cattrs.explain import (
    UnknownPredicateError,
    explain_structure_hook,
    explain_unstructure_hook,
)


class Foo:
    pass


class Bar:
    pass


def make_hook(log, name):
    def hook(v, cl):
        log.append((name, v, cl))
        return (name, v)

    return hook


def test_singledispatch_hook_takes_priority_over_predicate():
    """Concrete class hooks beat predicate hooks; no predicate is evaluated."""
    conv = BaseConverter()
    log = []
    concrete = make_hook(log, "concrete")
    predicate_hook = make_hook(log, "predicate")

    conv.register_structure_hook(Foo, concrete)
    conv.register_structure_hook_func(lambda t: t is Foo, predicate_hook)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "singledispatch"
    assert expl.hook is concrete
    assert expl.evaluations == []  # no predicate was evaluated
    assert log == []  # the conversion function was never called

    # Real structure uses the reported hook.
    assert conv.structure(1, Foo) == ("concrete", 1)
    assert log == [("concrete", 1, Foo)]


def test_predicate_hit_and_real_structure_matches():
    """A predicate hook is reported and used by real structure."""
    conv = BaseConverter()
    log = []
    hook = make_hook(log, "predicate")
    conv.register_structure_hook_func(lambda t: t is Foo, hook)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "predicate"
    assert expl.hook is hook
    assert expl.predicate is not None
    assert len(expl.evaluations) == 1
    assert expl.evaluations[0].result is True
    assert log == []  # the conversion function was never called

    # Real structure uses the reported hook.
    assert conv.structure(1, Foo) == ("predicate", 1)
    assert log == [("predicate", 1, Foo)]


def test_predicates_are_evaluated_once_in_registration_order():
    """Each predicate is evaluated at most once, in order, stopping at match."""
    conv = BaseConverter()
    log = []
    hook1 = make_hook(log, "h1")
    hook2 = make_hook(log, "h2")
    hook3 = make_hook(log, "h3")

    def p1(t):
        log.append("p1")
        return False

    def p2(t):
        log.append("p2")
        return True

    def p3(t):
        log.append("p3")
        return False

    conv.register_structure_hook_func(p1, hook1)
    conv.register_structure_hook_func(p2, hook2)
    conv.register_structure_hook_func(p3, hook3)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "predicate"
    assert expl.hook is hook2
    # p3 is checked first (last registered), then p2 matches; p1 is never
    # evaluated.
    assert [e.predicate for e in expl.evaluations] == [p3, p2]
    assert log == ["p3", "p2"]


def test_predicate_that_raises_is_skipped():
    """Predicates that raise are skipped, like in real dispatch."""
    conv = BaseConverter()
    log = []
    good = make_hook(log, "good")

    def raising(t):
        raise ValueError("boom")

    conv.register_structure_hook_func(lambda t: t is Foo, good)
    conv.register_structure_hook_func(raising, make_hook(log, "raising"))

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "predicate"
    assert expl.hook is good
    assert expl.evaluations[0].predicate is raising
    assert expl.evaluations[0].raised is True
    assert expl.evaluations[1].result is True


def test_factory_selection_is_distinguished_and_not_called():
    """A factory hit is reported as a factory, not a hook, and not called."""
    conv = BaseConverter()
    factory_calls = []
    hook_calls = []

    def hook(v, cl):
        hook_calls.append((v, cl))
        return ("fac", v)

    def factory(t):
        factory_calls.append(t)
        return hook

    conv.register_structure_hook_factory(lambda t: t is Foo, factory)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "factory"
    assert expl.factory is factory
    assert expl.is_generator is True
    assert expl.selected is factory
    assert factory_calls == []  # the factory was not called
    assert hook_calls == []  # the hook was not called

    # Real dispatch calls the factory and uses its hook.
    real_hook = conv.get_structure_hook(Foo)
    assert factory_calls == [Foo]
    assert real_hook is hook
    assert conv.structure(1, Foo) == ("fac", 1)
    assert hook_calls == [(1, Foo)]


def test_extended_factory_takes_converter():
    """Extended factories (taking a converter) are reported separately."""
    conv = BaseConverter()
    seen = []

    def factory(t, converter):
        seen.append((t, converter))
        return lambda v, _: v

    conv.register_structure_hook_factory(lambda t: t is Foo, factory)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "factory"
    assert expl.factory is factory
    assert expl.takes_converter is True
    assert seen == []


def test_fallback_selection():
    """With no match, the fallback factory is reported and not called."""
    conv = BaseConverter()
    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "fallback"
    assert expl.factory is conv._structure_func._fallback_factory
    assert expl.selected is conv._structure_func._fallback_factory


def test_generic_parameter_identity_is_preserved():
    """The exact type object (e.g. list[int]) is passed through unchanged."""
    conv = BaseConverter()
    seen = []
    target = list[int]

    def predicate(t):
        seen.append(t)
        return t is target

    conv.register_structure_hook_func(predicate, lambda v, _: v)

    expl = explain_structure_hook(conv, target)

    assert expl.typ is target
    assert seen[0] is target
    assert expl.kind == "predicate"


def test_unknown_predicate_can_be_rejected():
    """Unknown predicates (possible side effects) can optionally be refused."""
    conv = BaseConverter()
    side = []

    def bad_predicate(t):
        side.append(t)
        return True

    conv.register_structure_hook_func(bad_predicate, lambda v, _: v)

    with pytest.raises(UnknownPredicateError):
        explain_structure_hook(conv, Foo, reject_unknown_predicates=True)
    assert side == []  # the predicate was not called

    # By default it is evaluated.
    expl = explain_structure_hook(conv, Foo)
    assert expl.kind == "predicate"
    assert side == [Foo]


def test_explanation_does_not_modify_dispatch_cache():
    """Explaining leaves the converter dispatch cache untouched."""
    conv = BaseConverter()
    conv.register_structure_hook(Foo, lambda v, _: v)

    info_before = conv._structure_func.dispatch.cache_info()
    direct_before = dict(conv._structure_func._direct_dispatch)

    explain_structure_hook(conv, Foo)

    assert conv._structure_func.dispatch.cache_info() == info_before
    assert conv._structure_func._direct_dispatch == direct_before

    # Real dispatch still works exactly as before.
    assert conv.structure(1, Foo) == 1


def test_explanation_matches_real_dispatch_for_all_kinds():
    """The reported selection matches what real dispatch returns."""
    conv = BaseConverter()

    def hook(v, _):
        return ("hook", v)

    conv.register_structure_hook(Foo, hook)

    # singledispatch
    expl = explain_structure_hook(conv, Foo)
    assert expl.kind == "singledispatch"
    assert conv.get_structure_hook(Foo) is expl.hook

    # predicate
    conv2 = BaseConverter()

    def pred_hook(v, _):
        return ("pred", v)

    conv2.register_structure_hook_func(lambda t: t is Bar, pred_hook)
    expl2 = explain_structure_hook(conv2, Bar)
    assert expl2.kind == "predicate"
    assert conv2.get_structure_hook(Bar) is expl2.hook

    # factory
    conv3 = BaseConverter()

    def fac_hook(v, _):
        return ("fac", v)

    conv3.register_structure_hook_factory(lambda t: t is Bar, lambda t: fac_hook)
    expl3 = explain_structure_hook(conv3, Bar)
    assert expl3.kind == "factory"
    assert conv3.get_structure_hook(Bar) is expl3.factory(Bar)

    # fallback (use a custom fallback factory that returns a hook)
    def fallback_hook(v, _):
        return ("fallback", v)

    conv4 = BaseConverter(
        structure_fallback_factory=lambda t: lambda v, _: fallback_hook(v, t)
    )
    expl4 = explain_structure_hook(conv4, Foo)
    assert expl4.kind == "fallback"
    assert conv4.get_structure_hook(Foo)(1, Foo) == ("fallback", 1)


def test_unstructure_explanation():
    """Unstructure hooks are explained the same way."""
    conv = BaseConverter()
    log = []

    def hook(o):
        log.append(o)
        return ("un", o)

    conv.register_unstructure_hook(Foo, hook)

    expl = explain_unstructure_hook(conv, Foo)

    assert expl.kind == "singledispatch"
    assert expl.direction == "unstructure"
    assert expl.hook is hook
    assert log == []  # not called during explanation

    assert conv.unstructure(Foo())[0] == "un"
    assert len(log) == 1


def test_converter_method_delegates():
    """The converter methods delegate to the module functions."""
    conv = BaseConverter()
    conv.register_structure_hook(Foo, lambda v, _: v)
    conv.register_unstructure_hook(Foo, lambda o: o)

    expl = conv.explain_structure_hook(Foo)
    assert expl.kind == "singledispatch"

    expl_un = conv.explain_unstructure_hook(Foo)
    assert expl_un.kind == "singledispatch"
    assert expl_un.direction == "unstructure"


def test_direct_kind_for_generated_hook():
    """Generated hooks cached for a specific generic type show as direct."""
    conv = Converter()
    target = dict[str, int]
    conv.structure({"a": 1}, target)  # generate and cache the hook

    expl = explain_structure_hook(conv, target)

    assert expl.kind == "direct"
    assert expl.typ is target
    # The reported hook is the one real dispatch returns.
    assert conv.get_structure_hook(target) is expl.hook


def test_priority_example():
    """Concrete hook > predicate > factory, for the same type."""
    conv = BaseConverter()

    def concrete(v, _):
        return ("concrete", v)

    def pred_hook(v, _):
        return ("pred", v)

    def fac_hook(v, _):
        return ("fac", v)

    conv.register_structure_hook(Foo, concrete)
    conv.register_structure_hook_func(lambda t: t is Foo, pred_hook)
    conv.register_structure_hook_factory(lambda t: t is Foo, lambda t: fac_hook)

    expl = explain_structure_hook(conv, Foo)

    assert expl.kind == "singledispatch"
    assert expl.hook is concrete
    assert expl.evaluations == []  # predicates not even reached
