"""What each cluster task accepts, read off its implementation, and sweep validation.

Tasks read their parameters as ``p.get("key", default)`` (optional) or
``p["key"]`` (required), where ``p = ctx.params``, usually wrapped in the
conversion they apply (``int(...)``, ``mp.mpf(str(...))``, ``bool(...)``,
``tuple(int(s) for s in ...)``, ...). This module parses the task functions in
:mod:`virasoro_bootstrap.hpc.tasks` -- following ``p`` into the module-level
helpers it is handed to -- and recovers, per task, every accepted key, whether
it is required, what type the conversion implies, and the default expression.

That is the only list of task parameters: ``vb-submit`` validates a sweep
against it before writing anything (a misspelled key would otherwise be
ignored and its default used silently), ``vb-submit --list-tasks`` prints it,
and the task table in ``docs/CLUSTER.md`` is generated from it::

    python -m virasoro_bootstrap.hpc.params            # the markdown table

A task registered from another module is not analyzed: only its name is
checked.
"""
from __future__ import annotations

import ast
import difflib
import inspect
import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

#: Keys of a sweep file, and of its ``[resources]`` table (see ``render_sbatch``).
SWEEP_KEYS = frozenset({"name", "task", "resources", "params", "sweep", "sweep_rows"})
RESOURCE_KEYS = frozenset({"cpus_per_task", "time", "mem_per_cpu", "partition"})

_REAL_CONSTRUCTORS = {"mp.mpf", "mpmath.mpf", "mpf", "_mpf", "_central_charge", "_task_real_input",
                      "_validate_gap_bracket", "_validate_bisection_tolerance"}
_JOINT_VALIDATORS = {"_validate_gap_bracket"}
_ARGUMENT_CONVERTERS = {"_central_charge", "_task_real_input"}
_ARGUMENT_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
                       ast.Mult: operator.mul, ast.Div: operator.truediv}
_KIND_BY_CALL = {
    "int": "int",
    "_task_integer": "int",
    "_task_exchange_spin": "int",
    "_task_optional_integer": "int",
    "_task_spins": "list[int]",
    "_task_modular_spins": "list[int]",
    "_task_real_list": "list[real]",
    "_task_choice_list": "list",
    "_task_twist_gap": "twist_gap",
    "float": "real",
    "bool": "bool",
    "_task_boolean": "bool",
    "str": "str",
    "Path": "path",
    "dict": "mapping",
    "_merged_sdpb_options": "mapping",
    "tuple": "list",
    "list": "list",
}
_KIND_LABEL = {
    "int": "integer",
    "real": "number",
    "bool": "true/false",
    "str": "string",
    "path": "path",
    "mapping": "table",
    "list": "list",
    "list[int]": "list of integers",
    "list[real]": "list of numbers",
    "twist_gap": "none, match_gap, or number",
}


@dataclass(frozen=True)
class _IntegerRule:
    name: str
    minimum: int | None
    optional: bool
    unless_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ValueRule:
    validator: str
    before: tuple[Any, ...]
    after: tuple[Any, ...]
    keywords: tuple[tuple[str, Any], ...]
    optional: bool = False
    unless_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RuleExpression:
    operation: str
    arguments: tuple[Any, ...]


@dataclass(frozen=True)
class _JointRule:
    validator: str
    arguments: tuple[Any, ...]


@dataclass(frozen=True)
class ParamInfo:
    name: str
    kind: str | None        # key of _KIND_LABEL, or None when the use implies no type
    required: bool
    default: str | None     # the default expression as written, when optional
    integer_rules: tuple[_IntegerRule, ...] = ()
    value_rules: tuple[_ValueRule, ...] = ()


@dataclass(frozen=True)
class TaskSignature:
    task: str
    summary: str
    params: dict[str, ParamInfo]
    joint_rules: tuple[_JointRule, ...] = ()


# --- reading the task implementations ----------------------------------------

def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


class _Opaque(Exception):
    """``p`` is used in a way that could read keys we cannot see."""


@dataclass
class _Access:
    key: str
    node: ast.AST            # the p.get(...) Call or the p[...] Subscript
    required: bool
    default: ast.AST | None


class _FunctionReader:
    def __init__(self, module_functions: dict[str, ast.FunctionDef]):
        self.module_functions = module_functions
        self.accesses: list[_Access] = []
        self.parents: dict[ast.AST, ast.AST] = {}
        self.assigned_uses: dict[ast.AST, tuple[ast.Name, ...]] = {}
        self.assigned_values: dict[ast.Name, ast.AST] = {}
        self.joint_calls: list[ast.Call] = []
        self._seen: set[tuple[str, str]] = set()

    def read_task(self, fn_node: ast.FunctionDef) -> None:
        ctx_name = fn_node.args.args[0].arg if fn_node.args.args else "ctx"
        self._read(fn_node, aliases=set(), ctx_name=ctx_name)

    def _read(self, fn_node: ast.FunctionDef, *, aliases: set[str], ctx_name: str | None) -> None:
        for parent in ast.walk(fn_node):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent

        # A parameter may be saved once and validated in a later branch (the
        # optional modular spin list does this). Follow only names with one
        # assignment, so a later overwrite cannot imply a false constraint.
        stores: dict[str, int] = {}
        loads: dict[str, list[ast.Name]] = {}
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    stores[node.id] = stores.get(node.id, 0) + 1
                elif isinstance(node.ctx, ast.Load):
                    loads.setdefault(node.id, []).append(node)
        for node in ast.walk(fn_node):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and stores.get(node.targets[0].id) == 1):
                self.assigned_uses[node.value] = tuple(loads.get(node.targets[0].id, []))
                self.assigned_values.update((use, node.value) for use in self.assigned_uses[node.value])
            elif isinstance(node, ast.Call) and _dotted(node.func) in _JOINT_VALIDATORS:
                if node not in self.joint_calls:
                    self.joint_calls.append(node)

        def is_ctx_params(node: ast.AST) -> bool:
            return (
                ctx_name is not None
                and isinstance(node, ast.Attribute)
                and node.attr == "params"
                and isinstance(node.value, ast.Name)
                and node.value.id == ctx_name
            )

        aliases = set(aliases)
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Assign) and is_ctx_params(node.value):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        raise _Opaque("ctx.params assigned to a non-name")
                    aliases.add(target.id)

        for node in ast.walk(fn_node):
            if isinstance(node, ast.Name) and node.id in aliases and isinstance(node.ctx, ast.Load):
                self._use(node)
            elif is_ctx_params(node):
                parent = self.parents.get(node)
                if isinstance(parent, ast.Assign) and parent.value is node:
                    continue
                self._use(node)

    def _use(self, node: ast.AST) -> None:
        parent = self.parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            call = self.parents.get(parent)
            if parent.attr == "get" and isinstance(call, ast.Call) and call.func is parent:
                if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
                    raise _Opaque("p.get with a computed key")
                default = call.args[1] if len(call.args) > 1 else None
                self.accesses.append(_Access(call.args[0].value, call, False, default))
                return
            raise _Opaque(f"p.{parent.attr}")
        if isinstance(parent, ast.Subscript) and parent.value is node:
            key = parent.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(parent.ctx, ast.Load):
                self.accesses.append(_Access(key.value, parent, True, None))
                return
            raise _Opaque("p[...] with a computed key or a store")
        if isinstance(parent, ast.Compare) and node in parent.comparators:
            if isinstance(parent.left, ast.Constant) and isinstance(parent.left.value, str):
                self.accesses.append(_Access(parent.left.value, parent, False, None))
                return
            raise _Opaque("membership test with a computed key")
        if isinstance(parent, ast.Call) and node in parent.args:
            name = _dotted(parent.func)
            target = self.module_functions.get(name or "")
            if target is None:
                raise _Opaque(f"p passed to {name or 'an expression'}")
            index = parent.args.index(node)
            positional = [a.arg for a in target.args.args]
            if index >= len(positional):
                raise _Opaque(f"p passed as *args to {name}")
            self._follow(target, positional[index])
            return
        if isinstance(parent, ast.keyword) and isinstance(self.parents.get(parent), ast.Call):
            call = self.parents[parent]
            name = _dotted(call.func)
            target = self.module_functions.get(name or "")
            if target is None or parent.arg is None:
                raise _Opaque(f"p passed to {name or 'an expression'}")
            self._follow(target, parent.arg)
            return
        raise _Opaque(f"p used as {type(parent).__name__}")

    def _follow(self, fn_node: ast.FunctionDef, param_name: str) -> None:
        marker = (fn_node.name, param_name)
        if marker in self._seen:
            return
        self._seen.add(marker)
        self._read(fn_node, aliases={param_name}, ctx_name=None)

    # -- what a use implies ----------------------------------------------

    def rule_expression(self, node: ast.AST, seen=frozenset()):
        """Read exact defaults and related parameters without evaluating task code."""
        if node in seen:
            raise ValueError("cyclic task argument")
        seen = seen | {node}
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            pass
        for access in self.accesses:
            if access.node is node:
                default = self.rule_expression(access.default, seen) if access.default is not None else None
                return _RuleExpression("parameter", (access.key, access.required, default))
        if isinstance(node, ast.Name) and node in self.assigned_values:
            return self.rule_expression(self.assigned_values[node], seen)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARGUMENT_OPERATORS:
            return _RuleExpression(type(node.op).__name__,
                                   (self.rule_expression(node.left, seen), self.rule_expression(node.right, seen)))
        if isinstance(node, ast.Call) and _dotted(node.func) in _ARGUMENT_CONVERTERS and not node.keywords:
            return _RuleExpression("convert", (_dotted(node.func),
                                              *(self.rule_expression(arg, seen) for arg in node.args)))
        raise ValueError("unsupported task argument")

    def joint_rules(self) -> tuple[_JointRule, ...]:
        rules = []
        for call in self.joint_calls:
            if call.keywords:
                continue
            try:
                arguments = tuple(self.rule_expression(arg) for arg in call.args)
            except (ValueError, TypeError):
                continue
            rules.append(_JointRule(_dotted(call.func), arguments))
        return tuple(rules)

    def kind_of(self, node: ast.AST) -> str | None:
        parent = self.parents.get(node)
        if isinstance(parent, ast.Assign) and parent.value is node:
            if not self.value_rules_of(node):
                return None  # do not infer a type from unrelated downstream transforms
            return _narrowest([kind for use in self.assigned_uses.get(node, ())
                              if (kind := self.kind_of(use)) is not None])
        if isinstance(parent, ast.Call) and node in parent.args:
            name = _dotted(parent.func) or ""
            if name in _REAL_CONSTRUCTORS:
                return "real"
            if name == "str":
                outer = self.parents.get(parent)
                if isinstance(outer, ast.Call) and (_dotted(outer.func) or "") in _REAL_CONSTRUCTORS:
                    return "real"
                return "str"
            if name in _KIND_BY_CALL:
                return _KIND_BY_CALL[name]
            if isinstance(parent.func, ast.Attribute) and parent.func.attr == "get" and node is not parent.args[0]:
                return self.kind_of(parent)   # the default of an outer p.get
            return None
        if isinstance(parent, ast.BoolOp):
            return self.kind_of(parent)       # p.get(x) or DEFAULT
        if isinstance(parent, ast.Dict) and None in parent.keys:
            return "mapping"                  # {**(p.get(x) or {})}
        if isinstance(parent, ast.comprehension) and parent.iter is node:
            comp = self.parents.get(parent)
            element = getattr(comp, "elt", None)
            if isinstance(element, ast.Call):
                name = _dotted(element.func) or ""
                if name in ("int", "_task_integer"):
                    return "list[int]"
                if name in _REAL_CONSTRUCTORS or name == "float":
                    return "list[real]"
            return "list"
        return None

    def integer_rule_of(self, node: ast.AST) -> _IntegerRule | None:
        """Recover the runtime integer validator, including explicit minimums."""
        parent = self.parents.get(node)
        if isinstance(parent, ast.BoolOp):
            return self.integer_rule_of(parent)
        if not isinstance(parent, ast.Call) or node not in parent.args:
            return None
        if isinstance(parent.func, ast.Attribute) and parent.func.attr == "get" and node is not parent.args[0]:
            rule = self.integer_rule_of(parent)
            key = parent.args[0]
            if rule is not None and isinstance(key, ast.Constant) and isinstance(key.value, str):
                return _IntegerRule(rule.name, rule.minimum, rule.optional,
                                    (key.value, *rule.unless_present))
            return rule
        name = _dotted(parent.func)
        if name not in {"_task_integer", "_task_optional_integer"} or len(parent.args) != 2:
            return None
        label = parent.args[0]
        if not isinstance(label, ast.Constant) or not isinstance(label.value, str):
            return None
        minimum = None
        for keyword in parent.keywords:
            if keyword.arg == "minimum":
                if not isinstance(keyword.value, ast.Constant):
                    return None
                minimum = keyword.value.value
        return _IntegerRule(label.value, minimum, name == "_task_optional_integer")

    def value_rules_of(self, node: ast.AST) -> tuple[_ValueRule, ...]:
        """Recover the task's scalar and sequence validators."""
        parent = self.parents.get(node)
        if isinstance(parent, ast.Assign) and parent.value is node:
            return tuple(rule for use in self.assigned_uses.get(node, ())
                         for rule in self.value_rules_of(use))
        if isinstance(parent, ast.BoolOp):
            return self.value_rules_of(parent)
        if not isinstance(parent, ast.Call) or node not in parent.args:
            return ()
        if isinstance(parent.func, ast.Attribute) and parent.func.attr == "get" and node is not parent.args[0]:
            key = parent.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return tuple(_ValueRule(rule.validator, rule.before, rule.after, rule.keywords,
                                        rule.optional, (key.value, *rule.unless_present))
                             for rule in self.value_rules_of(parent))
            return ()
        name = _dotted(parent.func)
        if name not in {"_task_spins", "_task_modular_spins", "_task_exchange_spin",
                        "_task_choice_list", "_task_real_list", "_validate_bisection_tolerance"}:
            return ()
        index = parent.args.index(node)
        try:
            before = tuple(ast.literal_eval(arg) for arg in parent.args[:index])
            after = tuple(ast.literal_eval(arg) for arg in parent.args[index + 1:])
            keywords = tuple((kw.arg, ast.literal_eval(kw.value)) for kw in parent.keywords)
        except (ValueError, TypeError):
            return ()
        if any(key is None for key, _ in keywords):
            return ()
        optional = False
        if isinstance(node, ast.Name):
            child = parent
            while (ancestor := self.parents.get(child)) is not None:
                if isinstance(ancestor, ast.If):
                    test = ancestor.test
                    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                            and test.left.id == node.id and len(test.ops) == 1
                            and isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value is None):
                        optional |= ((isinstance(test.ops[0], ast.Is) and child in ancestor.orelse)
                                     or (isinstance(test.ops[0], ast.IsNot) and child in ancestor.body))
                child = ancestor
        return (_ValueRule(name, before, after, keywords, optional),)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    return str(value)


def _render_default(node: ast.AST | None) -> str | None:
    """The default expression of a ``p.get``, as a sweep author would write it."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return _toml_literal(node.value)
    if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Constant) for e in node.elts):
        return _toml_literal([e.value for e in node.elts])
    if isinstance(node, ast.Name):
        from . import tasks as _tasks

        value = getattr(_tasks, node.id, None)
        if isinstance(value, (str, int, float, bool)):
            return _toml_literal(value)
        if isinstance(value, dict):
            return f"tasks.{node.id}"
    if isinstance(node, ast.BoolOp):
        return _render_default(node.values[-1])
    if (isinstance(node, ast.Call) and _dotted(node.func) == "ctx.selector"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
        from . import tasks as _tasks

        return _toml_literal(getattr(_tasks, "DEFAULT_" + node.args[0].value.upper()))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        receiver = _dotted(node.func.value) or ""
        if node.args and isinstance(node.args[0], ast.Constant) and not any(
            word in receiver for word in ("preset", "profile", "prof")
        ):
            return f"same as {node.args[0].value}"
    text = ast.unparse(node)
    if any(word in text for word in ("preset", "profile")):
        return "from the lam profile"
    return text if len(text) <= 40 else "computed"


@lru_cache(maxsize=None)
def _module_functions(module_name: str) -> dict[str, ast.FunctionDef]:
    import importlib

    source = inspect.getsource(importlib.import_module(module_name))
    tree = ast.parse(source)
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _analyze(fn: Callable) -> tuple[dict[str, ParamInfo], tuple[_JointRule, ...]] | None:
    module_name = getattr(fn, "__module__", None)
    from . import tasks as _tasks

    if module_name != _tasks.__name__:
        return None
    functions = _module_functions(module_name)
    node = functions.get(fn.__name__)
    if node is None:
        return None
    reader = _FunctionReader(functions)
    try:
        reader.read_task(node)
    except _Opaque:
        return None
    merged: dict[str, dict[str, Any]] = {}
    for access in reader.accesses:
        entry = merged.setdefault(
            access.key, {"kinds": [], "required": False, "default": None, "optional_seen": False,
                         "integer_rules": [], "value_rules": []}
        )
        kind = reader.kind_of(access.node)
        if kind is not None and kind not in entry["kinds"]:
            entry["kinds"].append(kind)
        rule = reader.integer_rule_of(access.node)
        if rule is not None and rule not in entry["integer_rules"]:
            entry["integer_rules"].append(rule)
        for rule in reader.value_rules_of(access.node):
            if rule not in entry["value_rules"]:
                entry["value_rules"].append(rule)
        if access.required:
            entry["required"] = True
        else:
            entry["optional_seen"] = True
            if entry["default"] is None:
                default = access.default
                parent = reader.parents.get(access.node)
                if default is None and isinstance(parent, ast.BoolOp) and parent.values[0] is access.node:
                    default = parent.values[-1]   # p.get(x) or DEFAULT
                if (default is None and isinstance(parent, ast.Call)
                        and _dotted(parent.func) == "_merged_sdpb_options"
                        and len(parent.args) == 2 and parent.args[1] is access.node):
                    # _merged_sdpb_options(DEFAULTS, p.get(x)): the sweep's
                    # table is merged over DEFAULTS rather than replacing them.
                    base = _render_default(parent.args[0])
                    entry["default"] = "merged over " + (
                        base if base and base.startswith("tasks.") else "the task's defaults")
                    continue
                entry["default"] = _render_default(default)
    out = {}
    for key, entry in merged.items():
        # read as p[key] somewhere but guarded by a p.get(key) elsewhere is optional
        required = entry["required"] and not entry["optional_seen"]
        out[key] = ParamInfo(
            name=key,
            kind=_narrowest(entry["kinds"]),
            required=required,
            default=None if required else entry["default"],
            integer_rules=tuple(entry["integer_rules"]),
            value_rules=tuple(entry["value_rules"]),
        )
    return dict(sorted(out.items())), reader.joint_rules()


def _narrowest(kinds: list[str]) -> str | None:
    """One kind satisfying every use: the most specific of compatible kinds."""
    if not kinds:
        return None
    order = ["bool", "int", "list[int]", "real", "list[real]", "mapping", "path", "list", "twist_gap", "str"]
    ranked = sorted(kinds, key=order.index)
    return ranked[0]


def task_signature(name: str) -> TaskSignature | None:
    """The parameters task *name* accepts, or None when they cannot be read off."""
    from .tasks import get_task

    fn = get_task(name)
    analysis = _analyze(fn)
    if analysis is None:
        return None
    params, joint_rules = analysis
    doc = inspect.getdoc(fn) or ""
    summary = doc.strip().splitlines()[0].replace("``", "`") if doc.strip() else ""
    return TaskSignature(task=name, summary=summary, params=params, joint_rules=joint_rules)


# --- validating a sweep --------------------------------------------------------

def _resolve_rule_expression(value, parameters):
    if not isinstance(value, _RuleExpression):
        return value
    if value.operation == "parameter":
        key, required, default = value.arguments
        if key in parameters:
            return parameters[key]
        if required:
            raise KeyError(key)
        return _resolve_rule_expression(default, parameters)
    arguments = tuple(_resolve_rule_expression(arg, parameters) for arg in value.arguments)
    if value.operation == "convert":
        from . import tasks
        return getattr(tasks, arguments[0])(*arguments[1:])
    operation = next(function for kind, function in _ARGUMENT_OPERATORS.items()
                     if kind.__name__ == value.operation)
    return operation(*arguments)

def _check_value(key: str, value: Any, kind: str | None,
                 integer_rules: tuple[_IntegerRule, ...] = (),
                 value_rules: tuple[_ValueRule, ...] = ()) -> str | None:
    """A message when *value* cannot be what the task's conversion expects."""
    from .tasks import _task_integer_value, _task_real_input

    def is_real(v) -> bool:
        if not isinstance(v, (str, int, float)):
            return False
        try:
            _task_real_input(key, v)
            return True
        except (TypeError, ValueError, ArithmeticError):
            return False

    def is_int(v) -> bool:
        if not isinstance(v, (str, int, float)):
            return False
        try:
            _task_integer_value(key, v)
            return True
        except (TypeError, ValueError, ArithmeticError):
            return False

    if kind is None:
        return None
    if value is None and (integer_rules or value_rules) and all(
            rule.optional for rule in (*integer_rules, *value_rules)):
        return None
    label = _KIND_LABEL[kind]
    ok = {
        "int": is_int,
        "real": is_real,
        "bool": lambda v: isinstance(v, bool) or (type(v) is int and v in (0, 1)),
        "str": lambda v: isinstance(v, (str, int, float)) and not isinstance(v, bool),
        "path": lambda v: isinstance(v, str),
        "mapping": lambda v: isinstance(v, dict),
        "list": lambda v: isinstance(v, list),
        "list[int]": lambda v: isinstance(v, list) and all(is_int(e) for e in v),
        "list[real]": lambda v: isinstance(v, list) and all(is_real(e) for e in v),
        "twist_gap": lambda v: (isinstance(v, str) and v in _CHOICES["twist_gap"]) or is_real(v),
    }[kind](value)
    if ok:
        return None
    hint = ""
    if kind == "bool" and isinstance(value, str):
        hint = " (write true/false without quotes)"
    elif kind in ("real", "list[real]"):
        hint = " (finite, e.g. \"0.5\" or \"1/8\")"
    return f"{key} = {value!r} is not a valid {label}{hint}"


# Validate enumerated string parameters before task execution.
_CHOICES = {
    "backend": ("sdp", "pmp2sdp"),
    "bound_direction": ("upper", "lower"),
    "decide_by": ("status", "certificate"),
    "twist_gap": ("none", "match_gap"),
}


def _check_sdpb_options(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    errors = []
    for flag, setting in value.items():
        if setting is False or setting is None:
            continue  # runtime option normalization omits disabled entries
        if not str(flag).startswith("--"):
            errors.append(f"sdpb_options key {flag!r} is not an SDPB flag (flags start with --)")
        if isinstance(setting, (list, dict)):
            errors.append(f"sdpb_options {flag} = {setting!r}: a flag takes one value")
        name, equals, inline = str(flag).partition("=")
        if name in ("--precision", "--mpiProcs"):
            raw = inline if equals else setting
            if isinstance(raw, bool) or not _is_intlike(raw):
                errors.append(f"sdpb_options {name} needs an integer, got {raw!r}")
            elif int(str(raw)) < 1:
                errors.append(f"sdpb_options {name} must be >= 1, got {raw!r}")
    return errors


def _is_intlike(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        int(str(value))
        return True
    except ValueError:
        return False


def _suggest(key: str, known) -> str:
    close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def validate_sweep(config: dict, *, expand: Callable[[dict], list[dict]] | None = None) -> list[str]:
    """Every problem with a sweep config that is certain to break or be ignored.

    Returns messages (empty when the sweep is fine): an unknown task, sweep-file
    or ``[resources]`` keys it does not use, parameter keys the task never reads,
    missing required parameters, values its conversions cannot accept (non-numeric,
    NaN/inf, a quoted boolean, ...), a ``lam`` with no packaged profile, and a
    sweep that expands to no tasks.
    """
    from . import tasks as task_module
    from .tasks import LAMBDA_PRESETS, TASKS, _task_integer

    errors: list[str] = []

    def add(message: str) -> None:
        if message not in errors:
            errors.append(message)

    for key in config:
        if key not in SWEEP_KEYS:
            add(f"unknown sweep-file key {key!r}{_suggest(key, SWEEP_KEYS)}; "
                f"expected {', '.join(sorted(SWEEP_KEYS))}")
    resources = config.get("resources", {})
    if not isinstance(resources, dict):
        add("[resources] must be a table")
    else:
        for key in resources:
            if key not in RESOURCE_KEYS:
                add(f"unknown [resources] key {key!r}{_suggest(key, RESOURCE_KEYS)}; "
                    f"expected {', '.join(sorted(RESOURCE_KEYS))}")
        if "cpus_per_task" in resources:
            try:
                _task_integer("cpus_per_task", resources["cpus_per_task"], minimum=1)
            except (TypeError, ValueError, ArithmeticError) as exc:
                add(f"[resources] {exc}")

    task_name = config.get("task")
    signature = None
    if not isinstance(task_name, str) or not task_name:
        add("the sweep has no task = \"...\" (vb-submit --list-tasks shows them)")
    elif task_name not in TASKS:
        add(f"unknown task {task_name!r}{_suggest(task_name, TASKS)}; "
            "vb-submit --list-tasks shows the available tasks")
    else:
        signature = task_signature(task_name)

    for table in ("params", "sweep"):
        if table in config and not isinstance(config[table], dict):
            add(f"[{table}] must be a table")
    if any(e.endswith("must be a table") for e in errors):
        return errors

    sweep = config.get("sweep", {}) or {}
    empty = [key for key, values in sweep.items() if isinstance(values, list) and not values]
    for key in empty:
        add(f"[sweep] {key} is an empty list, so the sweep has no tasks")
    try:
        combos = (expand or _default_expand)(config)
    except ValueError as exc:
        add(str(exc))
        return errors
    if not combos and not empty:
        add("the sweep expands to no tasks")

    if signature is None:
        return errors
    accepted = signature.params

    # Every value a task could see, whether or not the sweep expands to any
    # task: the [params] values, each [[sweep_rows]] entry and each [sweep] value.
    rows = config.get("sweep_rows", [{}])
    base = dict(config.get("params", {}) or {})
    values: list[tuple[str, Any]] = list(base.items())
    for row in rows:
        values += list(row.items())
    for key, options in sweep.items():
        values += [(key, v) for v in (options if isinstance(options, list) else [options])]
    for key, value in values:
        info = accepted.get(key)
        if info is None:
            add(f"{task_name} has no parameter {key!r}{_suggest(key, accepted)}; "
                f"it would be ignored (vb-submit --list-tasks {task_name} lists them)")
            continue
        problem = _check_value(key, value, info.kind, info.integer_rules, info.value_rules)
        if problem:
            add(problem)
        if key == "sdpb_options":
            for message in _check_sdpb_options(value):
                add(message)
        choices = _CHOICES.get(key)
        if choices is not None and isinstance(value, str) and value not in choices and not (
                key == "twist_gap" and _check_value(key, value, "real") is None):
            add(f"{key} = {value!r} is not one of {', '.join(repr(c) for c in choices)}"
                + (" (or a number)" if key == "twist_gap" else "") + _suggest(value, choices))
        if key == "lam" and problem is None and int(value) not in LAMBDA_PRESETS:
            add(f"lam = {value} has no packaged profile; available: "
                f"{', '.join(str(k) for k in sorted(LAMBDA_PRESETS))}")
    for row in rows:
        present = set(base) | set(row) | set(sweep)
        for key, info in accepted.items():
            if info.required and key not in present:
                add(f"{task_name} needs {key!r} (set it in [params] or [sweep])")
    # Runtime constraints apply to the selected value. For p.get("four_point_dps", p.get("dps")),
    # an explicit four_point_dps overrides the fallback dps and its sphere floor.
    checked = set()
    for parameters in combos:
        for rule in signature.joint_rules:
            try:
                arguments = tuple(_resolve_rule_expression(arg, parameters) for arg in rule.arguments)
                getattr(task_module, rule.validator)(*arguments)
            except KeyError:
                continue  # Missing required parameters are reported above.
            except (TypeError, ValueError, ArithmeticError) as exc:
                add(str(exc))
        for key, value in parameters.items():
            info = accepted.get(key)
            if info is None or not (info.integer_rules or info.value_rules):
                continue
            rules = tuple(rule for rule in info.integer_rules
                          if not any(override in parameters for override in rule.unless_present))
            value_rules = tuple(rule for rule in info.value_rules
                                if not any(override in parameters for override in rule.unless_present))
            identity = value if isinstance(value, (str, int, float, type(None))) else id(value)
            marker = (key, type(value), identity, rules, value_rules)
            if not (rules or value_rules) or marker in checked:
                continue
            checked.add(marker)
            if _check_value(key, value, info.kind, info.integer_rules, info.value_rules) is not None:
                continue
            for rule in rules:
                if rule.optional and value is None:
                    continue
                try:
                    _task_integer(rule.name, value, minimum=rule.minimum)
                except (TypeError, ValueError, ArithmeticError) as exc:
                    add(f"{key} = {value!r}: {exc}")
            for rule in value_rules:
                if rule.optional and value is None:
                    continue
                try:
                    validator = getattr(task_module, rule.validator)
                    validator(*rule.before, value, *rule.after, **dict(rule.keywords))
                except (TypeError, ValueError, ArithmeticError) as exc:
                    add(f"{key} = {value!r}: {exc}")
    return errors


def _default_expand(config: dict) -> list[dict]:
    from .submit import expand_sweep

    return expand_sweep(config)


# --- describing tasks ------------------------------------------------------------

def _all_signatures() -> list[TaskSignature]:
    from .tasks import TASKS

    out = []
    for name in sorted(TASKS):
        sig = task_signature(name)
        if sig is not None:
            out.append(sig)
    return out


def _param_row(info: ParamInfo) -> tuple[str, str, str]:
    kind = _KIND_LABEL.get(info.kind, "any") if info.kind else "any"
    default = "required" if info.required else (info.default or "-")
    return info.name, kind, default


def describe_tasks(names: list[str] | None = None) -> str:
    """Plain-text listing of tasks and their parameters, for ``--list-tasks``."""
    from .tasks import TASKS, get_task

    lines: list[str] = []
    for name in names or sorted(TASKS):
        get_task(name)  # a clear error for an unknown name
        sig = task_signature(name)
        lines.append(name)
        if sig is None:
            lines.append("  (parameters not introspectable)")
            lines.append("")
            continue
        if sig.summary and not names:
            lines.append(f"  {sig.summary}")
        if names:
            doc = inspect.getdoc(get_task(name)) or ""
            doc = doc.replace("``", "`").replace(":func:", "").replace(":class:", "")
            lines += ["", *("  " + line if line else "" for line in doc.splitlines()), "", "  parameters:"]
        width = max((len(k) for k in sig.params), default=0)
        for info in sig.params.values():
            key, kind, default = _param_row(info)
            lines.append(f"    {key:<{width}}  {kind:<17} {default}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def markdown_table() -> str:
    """The task/parameter tables embedded in ``docs/CLUSTER.md``."""
    blocks = []
    for sig in _all_signatures():
        rows = [f"### `{sig.task}`", ""]
        if sig.summary:
            rows += [sig.summary, ""]
        rows += ["| Parameter | Type | Default |", "|---|---|---|"]
        for info in sig.params.values():
            key, kind, default = _param_row(info)
            prose = default in ("required", "-") or default.startswith(("from the ", "same as "))
            if default.startswith("merged over tasks."):
                cell = f"merged over `{default.removeprefix('merged over ')}`"
            else:
                cell = default if prose or default.startswith("merged over ") else f"`{default}`"
            rows.append(f"| `{key}` | {kind} | {cell.replace('|', chr(92) + '|')} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks) + "\n"


if __name__ == "__main__":  # pragma: no cover - documentation helper
    print(markdown_table(), end="")
