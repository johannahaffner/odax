from typing import Any

import jax.numpy as jnp
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair


def clip(expr: sympy.Expr, lo: sympy.Expr, hi: sympy.Expr) -> sympy.Expr:
    return sympy.Max(sympy.Min(expr, hi), lo)


def where(cond: sympy.Basic, x: sympy.Expr, y: sympy.Expr) -> sympy.Expr:
    return sympy.Piecewise((x, cond), (y, True))


def _piecewise(*pairs: tuple[Any, Any]) -> Any:
    result = pairs[-1][0]
    for expr_val, cond_val in reversed(pairs[:-1]):
        result = jnp.where(cond_val, expr_val, result)
    return result


def _expr_cond_pair(expr: Any, cond: Any) -> tuple[Any, Any]:
    return (expr, cond)


def _boolean_true() -> Any:
    return jnp.array(True)


def _boolean_false() -> Any:
    return jnp.array(False)


def _ite(cond: Any, true_val: Any, false_val: Any) -> Any:
    return jnp.where(cond, true_val, false_val)


piecewise_extra_funcs: dict[Any, Any] = {
    sympy.Piecewise: _piecewise,
    ExprCondPair: _expr_cond_pair,
    sympy.logic.boolalg.BooleanTrue: _boolean_true,
    sympy.logic.boolalg.BooleanFalse: _boolean_false,
    sympy.ITE: _ite,
}
