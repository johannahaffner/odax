import equinox as eqx
import sympy


class Reaction(eqx.Module):
    name: str
    rate: sympy.Expr
    stoichiometry: dict[str, int]

    def contribution(self, species_name: str) -> sympy.Expr:
        coeff = sympy.Integer(self.stoichiometry.get(species_name, 0))
        return coeff * self.rate
