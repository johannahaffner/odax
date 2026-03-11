import contextlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import sympy

from odax import Model, Parameter, Reaction, Species

from .helpers import hill_production, ode_systems


def _create_model(make_system, case):
    if make_system is hill_production:
        ctx = pytest.warns(UserWarning, match="does not seem to participate")
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        return Model(
            species=case.species,
            parameters=case.parameters,
            reactions=case.reactions,
            inputs=case.inputs,
        )


@pytest.mark.parametrize("make_system", ode_systems)
@pytest.mark.parametrize("space", ["log", "natural"])
def test_model_vector_field_matches_expected(make_system, space, getkey):
    case = make_system(space=space, trainable=True)
    model = _create_model(make_system, case)

    args = None if case.make_args is None else case.make_args()
    input_vals = (
        {}
        if args is None
        else {name: path.evaluate(jnp.array(0.0)) for name, path in args.items()}
    )

    y = {
        s.name: jax.random.uniform(getkey(), minval=0.1, maxval=5.0)
        for s in case.species
        if s.expr is None
    }
    result = model(t=jnp.array(0.0), y=y, args=args)

    params = {p.name: p.value for p in case.parameters}
    expected = case.expected_vector_field(**y, **params, **input_vals)

    for name in result:
        assert jnp.allclose(result[name], jnp.array(expected[name]))


def test_model_raises_on_duplicate_names():
    x = Species(name="x")
    k1 = Parameter(name="k", value=0.5, trainable=False, space="natural")
    k2 = Parameter(name="k", value=0.1, trainable=False, space="natural")
    rxn = Reaction(name="decay", rate=k1.sym * x.sym, stoichiometry={"x": -1})
    with pytest.raises(
        ValueError, match="Parameter name 'k' duplicates existing Parameter"
    ):
        Model(species=[x], parameters=[k1, k2], reactions=[rxn], inputs=[])


def test_model_raises_on_species_parameter_name_collision():
    x = Species(name="x")
    k = Parameter(name="x", value=0.5, trainable=False, space="natural")
    rxn = Reaction(name="decay", rate=k.sym * x.sym, stoichiometry={"x": -1})
    with pytest.raises(
        ValueError, match="Parameter name 'x' duplicates existing Species"
    ):
        Model(species=[x], parameters=[k], reactions=[rxn], inputs=[])


def test_model_raises_on_unknown_symbol_in_rate():
    x = Species(name="x")
    k = Parameter(name="k", value=0.5, trainable=False, space="natural")
    mystery = sympy.Symbol("z")
    rxn = Reaction(name="decay", rate=mystery * x.sym, stoichiometry={"x": -1})
    with pytest.raises(ValueError, match="unknown symbol"):
        Model(species=[x], parameters=[k], reactions=[rxn], inputs=[])


def test_algebraic_species_raises_if_stoichiometric_participant():
    x = Species(name="x")
    k = Parameter(name="k", value=0.5, trainable=False, space="natural")
    inactive = Species(name="inactive", expr=sympy.Integer(1) - x.sym)
    rxn = Reaction(
        name="rxn",
        rate=k.sym * x.sym,
        stoichiometry={"x": 1, "inactive": -1},
    )
    with pytest.raises(ValueError, match="Algebraic"):
        Model(species=[x, inactive], parameters=[k], reactions=[rxn])


def test_algebraic_species_forward_reference_raises():
    x = Species(name="x")
    b = Species(name="b", expr=sympy.Symbol("a") + x.sym)
    a = Species(name="a", expr=x.sym)
    k = Parameter(name="k", value=0.5, trainable=False, space="natural")
    rxn = Reaction(name="rxn", rate=k.sym * x.sym, stoichiometry={"x": -1})
    with pytest.raises(ValueError, match="Unknown symbol"):
        Model(species=[x, b, a], parameters=[k], reactions=[rxn])


@pytest.mark.parametrize("make_system", ode_systems)
@pytest.mark.parametrize("space", ["log", "natural"])
@pytest.mark.parametrize("trainable", [True, False])
def test_trainable_partition(make_system, space, trainable, getkey):
    case = make_system(space=space, trainable=trainable)
    model = _create_model(make_system, case)

    dynamic, static = eqx.partition(model, model.trainable_filter, replace=None)
    for param in case.parameters:
        if not param.trainable:
            leaves = jax.tree_util.tree_leaves(dynamic.parameters[param.name].param)
            assert all(leaf is None for leaf in leaves)
