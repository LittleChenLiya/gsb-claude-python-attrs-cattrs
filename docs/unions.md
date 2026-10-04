# Handling Unions

_cattrs_ is able to handle simple unions of _attrs_ classes and dataclasses [automatically](#default-union-strategy).
More complex cases require converter customization (since there are many ways of handling unions).

_cattrs_ also comes with a number of optional strategies to help handle unions:

- [tagged unions strategy](strategies.md#tagged-unions-strategy) mentioned below
- [union passthrough strategy](strategies.md#union-passthrough), which is preapplied to all the [preconfigured](preconf.md) converters

## Default Union Strategy

For convenience, _cattrs_ includes a default union structuring strategy which is a little more opinionated.

Given a union of several _attrs_ classes and/or dataclasses, the default union strategy will attempt to handle it in several ways.

First, it will look for `Literal` fields.
If _all members_ of the union contain a literal field, _cattrs_ will generate a disambiguation function based on the field.

```python
from typing import Literal

@define
class ClassA:
    field_one: Literal["one"]

@define
class ClassB:
    field_one: Literal["two"] = "two"
```

In this case, a payload containing `{"field_one": "one"}` will produce an instance of `ClassA`.

````{note}
The following snippet can be used to disable the use of literal fields, restoring legacy behavior.

```python
from functools import partial
from cattrs.disambiguators import is_supported_union

converter.register_structure_hook_factory(
    is_supported_union,
    partial(converter._gen_attrs_union_structure, use_literals=False),
)
```

````

If there are no appropriate fields, the strategy will examine the classes for **unique required fields**.

So, given a union of `ClassA` and `ClassB`:

```python
@define
class ClassA:
    field_one: str
    field_with_default: str = "a default"

@define
class ClassB:
    field_two: str
```

the strategy will determine that if a payload contains the key `field_one` it should be handled as `ClassA`, and if it contains the key `field_two` it should be handled as `ClassB`.
The field `field_with_default` will not be considered since it has a default value, so it gets treated as optional.

```{versionchanged} 23.2.0
Literals can now be potentially used to disambiguate.
```

```{versionchanged} 24.1.0
Dataclasses are now supported in addition to _attrs_ classes.
```

## Unstructuring Unions with Extra Metadata

```{note}
_cattrs_ comes with the [tagged unions strategy](strategies.md#tagged-unions-strategy) for handling this exact use-case since version 23.1.
The example below has been left here for educational purposes, but you should prefer the strategy.
```

Let's assume a simple scenario of two classes, `ClassA` and `ClassB`, both
of which have no distinct fields and so cannot be used automatically with
_cattrs_.

```python
@define
class ClassA:
    a_string: str

@define
class ClassB:
    a_string: str
```

A naive approach to unstructuring either of these would yield identical
dictionaries, and not enough information to restructure the classes.

```python
>>> converter.unstructure(ClassA("test"))
{'a_string': 'test'}  # Is this ClassA or ClassB? Who knows!
```

What we can do is ensure some extra information is present in the
unstructured data, and then use that information to help structure later.

First, we register an unstructure hook for the `Union[ClassA, ClassB]` type.

```python
>>> converter.register_unstructure_hook(
...     Union[ClassA, ClassB],
...     lambda o: {"_type": type(o).__name__,  **converter.unstructure(o)}
... )
>>> converter.unstructure(ClassA("test"), unstructure_as=Union[ClassA, ClassB])
{'_type': 'ClassA', 'a_string': 'test'}
```

Note that when unstructuring, we had to provide the `unstructure_as` parameter
or _cattrs_ would have just applied the usual unstructuring rules to `ClassA`,
instead of our special union hook.

Now that the unstructured data contains some information, we can create a
structuring hook to put it to use:

```python
>>> converter.register_structure_hook(
...     Union[ClassA, ClassB],
...     lambda o, _: converter.structure(o, ClassA if o["_type"] == "ClassA" else ClassB)
... )
>>> converter.structure({"_type": "ClassA", "a_string": "test"}, Union[ClassA, ClassB])
ClassA(a_string='test')
```

## Diagnostic Union Structuring

When a union fails to structure, the default strategy only surfaces the error
of the single branch it selected, which can leave you guessing why the other
branches did not match. `structure_diagnostic` is an explicit, opt-in entry
point (available on all converters and as `cattrs.structure_diagnostic`) which
instead tries every branch and reports each branch's reason for failing.
The default `structure` behavior is unchanged.

```python
>>> from typing import Union
>>> from attrs import define
>>> from cattrs import Converter, UnionStructureError

>>> @define
... class A:
...     x: int

>>> @define
... class B:
...     y: str

>>> converter = Converter()
>>> try:
...     converter.structure_diagnostic({"x": "bad"}, Union[A, B])
... except UnionStructureError as exc:
...     print(len(exc.exceptions))  # one error per branch, in declaration order
2
```

Here the report contains branch `A`'s error (`x` failed to structure) *and*
branch `B`'s error (`y` is missing), with nested field errors
(`ClassValidationError`/`IterableValidationError` and their notes) kept intact,
so `cattrs.transform_error` still produces full paths for each branch.

### Branch Selection

- Branches are attempted in the order they are declared in the union, and each
  branch is attempted **at most once** per call.
- Each branch is structured through the converter's normal hook dispatch, so
  custom hooks registered for branch types keep their existing registration
  priority. A hook registered for the exact union itself also takes priority,
  exactly as with `structure`, and is called at most once.
- If exactly one branch succeeds, its structured value is returned directly -
  the original converted type, not a wrapper.
- If no branch succeeds, a {class}`UnionStructureError <cattrs.errors.UnionStructureError>`
  is raised; its `exceptions` hold every branch's error in registration order
  (not just the last branch's), and `branches` holds the matching branch types.
- If several branches succeed, the `strict_ambiguity` parameter (on by default)
  applies: a {class}`UnionAmbiguityError <cattrs.errors.UnionAmbiguityError>`
  is raised as an ambiguity report, carrying all successful `(branch, value)`
  pairs. With `strict_ambiguity=False`, the first successful branch in
  registration order wins.

### Side-effect Boundary

Diagnostic structuring is not side-effect free: to detect ambiguity, **all**
branches are attempted even after one succeeds, so a structuring hook with
side effects observes exactly one call per branch per invocation. Failures
only ever raise - partially-structured ("half-built") results are never
returned to the caller; the ambiguity report only contains fully-structured
values.

The tests for this feature can be run locally with:

```bash
python -m pytest tests/test_structure_diagnostic.py -q
```
