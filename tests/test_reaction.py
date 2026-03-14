import jax
import jax.numpy as jnp
import pytest
import sympy
import sympy2jax as s2j

from odax._sympy_utils import piecewise_extra_funcs

from .helpers import ode_systems


@pytest.mark.parametrize("make_system", ode_systems)
def test_reaction_stoichiometry_values_are_ints(make_system):
    case = make_system(space="log", trainable=True)
    for rxn in case.reactions:
        for coeff in rxn.stoichiometry.values():
            assert isinstance(coeff, int)


@pytest.mark.parametrize("make_system", ode_systems)
def test_reaction_contribution_assembles_derivatives(make_system):
    case = make_system(space="log", trainable=True)  # These options do not matter here
    subs = [(s.sym, s.expr) for s in case.species if s.expr is not None]
    for species in [s for s in case.species if s.expr is None]:
        derivative = sympy.Integer(0)
        for rxn in case.reactions:
            derivative += rxn.contribution(species.name)
        derivative = derivative.subs(subs)
        expected = case.expected_derivatives[species.name]
        assert sympy.simplify(derivative - expected) == 0


@pytest.mark.parametrize("make_system", ode_systems)
def test_sympy2jax_matches_sympy_and_expected_vector_field(make_system, getkey):
    case = make_system(space="natural", trainable=True)

    numeric = {}
    for s in case.species:
        if s.expr is None:
            numeric[s.name] = jax.random.uniform(getkey(), minval=0.1, maxval=5.0)
    for p in case.parameters:
        numeric[p.name] = p.value
    if case.make_args is not None:
        for name, path in case.make_args().items():
            numeric[name] = path.evaluate(jnp.array(0.5))

    expected_vf = case.expected_vector_field(**numeric)
    interm_subs = [(s.sym, s.expr) for s in case.species if s.expr is not None]
    numeric_subs = {sympy.Symbol(k): float(v) for k, v in numeric.items()}

    for species in [s for s in case.species if s.expr is None]:
        derivative = sympy.Integer(0)
        for rxn in case.reactions:
            derivative += rxn.contribution(species.name)
        derivative = derivative.subs(interm_subs)

        sympy_result = jnp.array(float(derivative.subs(numeric_subs)))  # pyright: ignore[reportCallIssue, reportArgumentType]
        s2j_result = s2j.SymbolicModule(derivative, extra_funcs=piecewise_extra_funcs)(
            **numeric
        )
        vf_result = jnp.array(expected_vf[species.name])

        assert jnp.allclose(sympy_result, vf_result)
        assert jnp.allclose(s2j_result, vf_result)
