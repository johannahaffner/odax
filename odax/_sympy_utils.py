import sympy


def clip(expr: sympy.Expr, lo: sympy.Expr, hi: sympy.Expr) -> sympy.Expr:
    return sympy.Max(sympy.Min(expr, hi), lo)
