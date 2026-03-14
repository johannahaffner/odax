import warnings
from collections.abc import Mapping
from typing import cast

import equinox as eqx
import jax
import paramax as px
import sympy
import sympy2jax
from jaxtyping import Array, Float, Shaped

from ._input import AbstractPath, Input
from ._parameter import Parameter
from ._reaction import Reaction
from ._species import Species


class Model(eqx.Module):
    parameters: dict[str, Parameter]
    inputs: tuple[Input, ...]
    derivative_modules: dict[str, sympy2jax.SymbolicModule]

    def __init__(
        self,
        species: list[Species],
        parameters: list[Parameter],
        reactions: list[Reaction],
        inputs: list[Input] | None = None,
    ) -> None:
        ode_species = [s for s in species if s.expr is None]
        algebraic_species = [s for s in species if s.expr is not None]

        seen: dict[str, str] = {}  # name -> entity type

        # Pass 1: register ODE species, parameters, inputs
        for label, group in [
            ("Species", ode_species),
            ("Parameter", parameters),
            ("Input", inputs or []),
        ]:
            for item in group:
                if item.name in seen:
                    raise ValueError(
                        f"{label} name '{item.name}' duplicates existing "
                        f"{seen[item.name]} with the same name. Names should be unique."
                    )
                seen[item.name] = label

        # Pass 2: validate and register algebraic species in list order
        for s in algebraic_species:
            for sym in cast(set[sympy.Symbol], s.expr.free_symbols):  # type: ignore[union-attr]
                if sym.name not in seen:
                    raise ValueError(
                        f"Unknown symbol '{sym.name}' in algebraic species '{s.name}'."
                    )
            if s.name in seen:
                raise ValueError(
                    f"Species name '{s.name}' duplicates existing "
                    f"{seen[s.name]} with the same name. Names should be unique."
                )
            seen[s.name] = "Species"

        # Validate reactions: algebraic species cannot be stoichiometric participants
        algebraic_names = {s.name for s in algebraic_species}
        for rxn in reactions:
            for name in rxn.stoichiometry:
                if name in algebraic_names:
                    raise ValueError(
                        f"Algebraic species '{name}' cannot appear as a stoichiometric "
                        f"participant in reaction '{rxn.name}'."
                    )
            for sym in cast(set[sympy.Symbol], rxn.rate.free_symbols):
                if sym.name not in seen:
                    raise ValueError(
                        f"Unknown symbol '{sym.name}' in rate of reaction '{rxn.name}'."
                    )

        self.parameters = {p.name: p for p in parameters}
        self.inputs = tuple(inputs or [])

        subs = [(s.sym, s.expr) for s in algebraic_species if s.expr is not None]
        derivative_modules = {}
        for s in ode_species:
            expr = sympy.Integer(0)
            for rxn in reactions:
                expr += rxn.contribution(s.name)
            expr = expr.subs(subs)
            if expr.is_zero:
                warnings.warn(
                    f"Species '{s.name}' does not seem to participate in any "
                    "reactions. Its ODE will always return zero, degrades the quality "
                    "of the Jacobian of the vector field. This in turn may not play "
                    "nicely with stiff ODE solvers.",
                    stacklevel=2,
                )
            derivative_modules[s.name] = sympy2jax.SymbolicModule(expr)
        self.derivative_modules = derivative_modules

    def __call__(
        self,
        t: Float[Array, ""],
        y: dict[str, Float[Array, ""]],
        args: Mapping[str, AbstractPath] | None,
    ) -> dict[str, Shaped[Array, ""]]:
        params = {name: px.unwrap(p.param) for name, p in self.parameters.items()}
        input_vals = (
            {}
            if args is None
            else {name: path.evaluate(t) for name, path in args.items()}
        )
        combined = {**y, **params, **input_vals}
        return {
            name: module(**combined) for name, module in self.derivative_modules.items()
        }


def trainable_filter(model: Model):
    spec = jax.tree_util.tree_map(lambda _: False, model)
    # JAX sorts dict keys when flattening pytrees, so iterate in sorted order to match
    # the tree structure of spec.
    return eqx.tree_at(
        lambda m: tuple(m.parameters.values()),
        spec,
        replace=tuple(
            eqx.is_array
            if isinstance(model.parameters[k].param, px.Parameterize)
            else False
            for k in sorted(model.parameters)
        ),
    )
