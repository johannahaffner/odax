import equinox as eqx
import sympy


class Species(eqx.Module):
    name: str
    expr: sympy.Expr | None = None

    @property
    def sym(self) -> sympy.Symbol:
        return sympy.Symbol(self.name)
