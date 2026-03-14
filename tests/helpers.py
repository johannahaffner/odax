from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple

import jax.numpy as jnp
import sympy

import odax


class ODETestCase(NamedTuple):
    species: list[odax.Species]
    parameters: list[odax.Parameter]
    reactions: list[odax.Reaction]
    expected_derivatives: dict[str, sympy.Expr]
    expected_vector_field: Callable[..., dict[str, Any]]
    inputs: list[odax.Input]
    make_args: Callable[[], Mapping[str, Any]] | None


ode_systems: list = []


def _register(fn):  # noqa: ANN001, ANN202
    ode_systems.append(fn)
    return fn


@_register
def simple_decay(*, space: Literal["log", "natural"], trainable: bool) -> ODETestCase:
    """dx/dt = -k * x"""
    x = odax.Species(name="x")
    k = odax.Parameter(name="k", value=0.5, trainable=trainable, space=space)

    rxn = odax.Reaction(
        name="decay",
        rate=k.sym * x.sym,
        stoichiometry={"x": -1},
    )

    x_sym, k_sym = sympy.symbols("x k")
    return ODETestCase(
        species=[x],
        parameters=[k],
        reactions=[rxn],
        expected_derivatives={"x": -k_sym * x_sym},
        expected_vector_field=lambda *, x, k: {"x": -k * x},
        inputs=[],
        make_args=None,
    )


@_register
def production_and_decay_fixed_decay(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = alpha - k * x, where k is always non-trainable."""
    x = odax.Species(name="x")
    alpha = odax.Parameter(name="alpha", value=1.0, trainable=trainable, space=space)
    k = odax.Parameter(name="k", value=0.5, trainable=False, space=space)

    production = odax.Reaction(
        name="production",
        rate=alpha.sym,
        stoichiometry={"x": 1},
    )
    decay = odax.Reaction(
        name="decay",
        rate=k.sym * x.sym,
        stoichiometry={"x": -1},
    )

    x_sym, alpha_sym, k_sym = sympy.symbols("x alpha k")
    return ODETestCase(
        species=[x],
        parameters=[alpha, k],
        reactions=[production, decay],
        expected_derivatives={"x": alpha_sym - k_sym * x_sym},
        expected_vector_field=lambda *, x, alpha, k: {"x": alpha - k * x},
        inputs=[],
        make_args=None,
    )


@_register
def production_and_decay(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = alpha - k * x"""
    x = odax.Species(name="x")
    alpha = odax.Parameter(name="alpha", value=1.0, trainable=trainable, space=space)
    k = odax.Parameter(name="k", value=0.5, trainable=trainable, space=space)

    production = odax.Reaction(
        name="production",
        rate=alpha.sym,
        stoichiometry={"x": 1},
    )
    decay = odax.Reaction(
        name="decay",
        rate=k.sym * x.sym,
        stoichiometry={"x": -1},
    )

    x_sym, alpha_sym, k_sym = sympy.symbols("x alpha k")
    return ODETestCase(
        species=[x],
        parameters=[alpha, k],
        reactions=[production, decay],
        expected_derivatives={"x": alpha_sym - k_sym * x_sym},
        expected_vector_field=lambda *, x, alpha, k: {"x": alpha - k * x},
        inputs=[],
        make_args=None,
    )


@_register
def two_species_conversion(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = -k * x, dy/dt = +k * x"""
    x = odax.Species(name="x")
    y = odax.Species(name="y")
    k = odax.Parameter(name="k", value=0.3, trainable=trainable, space=space)

    rxn = odax.Reaction(
        name="conversion",
        rate=k.sym * x.sym,
        stoichiometry={"x": -1, "y": 1},
    )

    x_sym, k_sym = sympy.symbols("x k")
    return ODETestCase(
        species=[x, y],
        parameters=[k],
        reactions=[rxn],
        expected_derivatives={
            "x": -k_sym * x_sym,
            "y": k_sym * x_sym,
        },
        expected_vector_field=lambda *, x, y, k: {
            "x": -k * x,
            "y": k * x,
        },
        inputs=[],
        make_args=None,
    )


@_register
def michaelis_menten(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = -Vmax * x / (Km + x)"""
    x = odax.Species(name="x")
    vmax = odax.Parameter(name="Vmax", value=1.0, trainable=trainable, space=space)
    km = odax.Parameter(name="Km", value=0.5, trainable=trainable, space=space)

    rxn = odax.Reaction(
        name="consumption",
        rate=vmax.sym * x.sym / (km.sym + x.sym),
        stoichiometry={"x": -1},
    )

    x_sym, vmax_sym, km_sym = sympy.symbols("x Vmax Km")
    return ODETestCase(
        species=[x],
        parameters=[vmax, km],
        reactions=[rxn],
        expected_derivatives={
            "x": -vmax_sym * x_sym / (km_sym + x_sym),
        },
        expected_vector_field=lambda *, x, Vmax, Km: {
            "x": -Vmax * x / (Km + x),
        },
        inputs=[],
        make_args=None,
    )


@_register
def reversible_binding(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """
    dx/dt = -kon * x * y + koff * z
    dy/dt = -kon * x * y + koff * z
    dz/dt = +kon * x * y - koff * z
    """
    x = odax.Species(name="x")
    y = odax.Species(name="y")
    z = odax.Species(name="z")
    kon = odax.Parameter(name="kon", value=1.0, trainable=trainable, space=space)
    koff = odax.Parameter(name="koff", value=0.1, trainable=trainable, space=space)

    bind = odax.Reaction(
        name="binding",
        rate=kon.sym * x.sym * y.sym,
        stoichiometry={"x": -1, "y": -1, "z": 1},
    )
    unbind = odax.Reaction(
        name="unbinding",
        rate=koff.sym * z.sym,
        stoichiometry={"x": 1, "y": 1, "z": -1},
    )

    x_sym, y_sym, z_sym = sympy.symbols("x y z")
    kon_sym, koff_sym = sympy.symbols("kon koff")
    return ODETestCase(
        species=[x, y, z],
        parameters=[kon, koff],
        reactions=[bind, unbind],
        expected_derivatives={
            "x": -kon_sym * x_sym * y_sym + koff_sym * z_sym,
            "y": -kon_sym * x_sym * y_sym + koff_sym * z_sym,
            "z": kon_sym * x_sym * y_sym - koff_sym * z_sym,
        },
        expected_vector_field=lambda *, x, y, z, kon, koff: {
            "x": -kon * x * y + koff * z,
            "y": -kon * x * y + koff * z,
            "z": kon * x * y - koff * z,
        },
        inputs=[],
        make_args=None,
    )


@_register
def hill_production(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = Vmax * s^n / (K^n + s^n) - k * x"""
    s = odax.Species(name="s")
    x = odax.Species(name="x")
    vmax = odax.Parameter(name="Vmax", value=1.0, trainable=trainable, space=space)
    K = odax.Parameter(name="K", value=0.5, trainable=trainable, space=space)
    n = odax.Parameter(name="n", value=2.0, trainable=trainable, space=space)
    k = odax.Parameter(name="k", value=0.1, trainable=trainable, space=space)

    production = odax.Reaction(
        name="hill_production",
        rate=vmax.sym * s.sym**n.sym / (K.sym**n.sym + s.sym**n.sym),
        stoichiometry={"x": 1},
    )
    decay = odax.Reaction(
        name="decay",
        rate=k.sym * x.sym,
        stoichiometry={"x": -1},
    )

    s_sym, x_sym = sympy.symbols("s x")
    vmax_sym, K_sym, n_sym, k_sym = sympy.symbols("Vmax K n k")
    return ODETestCase(
        species=[s, x],
        parameters=[vmax, K, n, k],
        reactions=[production, decay],
        expected_derivatives={
            "s": sympy.Integer(0),
            "x": (
                vmax_sym * s_sym**n_sym / (K_sym**n_sym + s_sym**n_sym) - k_sym * x_sym
            ),
        },
        expected_vector_field=lambda *, s, x, Vmax, K, n, k: {
            "s": 0.0 * s,
            "x": Vmax * s**n / (K**n + s**n) - k * x,
        },
        inputs=[],
        make_args=None,
    )


@_register
def dimerisation(*, space: Literal["log", "natural"], trainable: bool) -> ODETestCase:
    """dx/dt = -2 * k * x**2, dy/dt = k * x**2"""
    x = odax.Species(name="x")
    y = odax.Species(name="y")
    k = odax.Parameter(name="k", value=0.1, trainable=trainable, space=space)

    rxn = odax.Reaction(
        name="dimerisation",
        rate=k.sym * x.sym**2,
        stoichiometry={"x": -2, "y": 1},
    )

    x_sym, k_sym = sympy.symbols("x k")
    return ODETestCase(
        species=[x, y],
        parameters=[k],
        reactions=[rxn],
        expected_derivatives={
            "x": -2 * k_sym * x_sym**2,
            "y": k_sym * x_sym**2,
        },
        expected_vector_field=lambda *, x, y, k: {
            "x": -2 * k * x**2,
            "y": k * x**2,
        },
        inputs=[],
        make_args=None,
    )


@_register
def forced_production(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = u(t) - k*x, where u is an external forcing input."""
    import diffrax

    u = odax.Input(name="u")
    x = odax.Species(name="x")
    k = odax.Parameter(name="k", value=0.5, trainable=trainable, space=space)

    production = odax.Reaction(name="production", rate=u.sym, stoichiometry={"x": 1})
    decay = odax.Reaction(name="decay", rate=k.sym * x.sym, stoichiometry={"x": -1})

    x_sym, k_sym, u_sym = sympy.symbols("x k u")
    return ODETestCase(
        species=[x],
        parameters=[k],
        reactions=[production, decay],
        expected_derivatives={"x": u_sym - k_sym * x_sym},
        expected_vector_field=lambda *, x, k, u: {"x": u - k * x},
        inputs=[u],
        make_args=lambda: {
            "u": diffrax.LinearInterpolation(
                ts=jnp.array([0.0, 1.0]), ys=jnp.array([2.0, 2.0])
            )
        },
    )


@_register
def clipped_decay(*, space: Literal["log", "natural"], trainable: bool) -> ODETestCase:
    """dx/dt = -k * clip(x, lo, hi), using sympy.Max(sympy.Min(...)) for clip."""
    x = odax.Species(name="x")
    k = odax.Parameter(name="k", value=0.5, trainable=trainable, space=space)
    lo = odax.Parameter(name="lo", value=0.1, trainable=False, space="natural")
    hi = odax.Parameter(name="hi", value=5.0, trainable=False, space="natural")

    clipped = odax.clip(x.sym, lo.sym, hi.sym)
    rxn = odax.Reaction(
        name="clipped_decay",
        rate=k.sym * clipped,
        stoichiometry={"x": -1},
    )

    x_sym, k_sym, lo_sym, hi_sym = sympy.symbols("x k lo hi")
    return ODETestCase(
        species=[x],
        parameters=[k, lo, hi],
        reactions=[rxn],
        expected_derivatives={
            "x": -k_sym * sympy.Max(sympy.Min(x_sym, hi_sym), lo_sym)
        },
        expected_vector_field=lambda *, x, k, lo, hi: {
            "x": -k * jnp.clip(x, lo, hi),
        },
        inputs=[],
        make_args=None,
    )


@_register
def fractional_activation(
    *, space: Literal["log", "natural"], trainable: bool
) -> ODETestCase:
    """dx/dt = k_on * (1 - x) - k_off * x, using an algebraic species for (1 - x)."""
    x = odax.Species(name="x")
    k_on = odax.Parameter(name="k_on", value=1.0, trainable=trainable, space=space)
    k_off = odax.Parameter(name="k_off", value=0.5, trainable=trainable, space=space)
    inactive = odax.Species(name="inactive", expr=sympy.Integer(1) - x.sym)

    activation = odax.Reaction(
        name="activation",
        rate=k_on.sym * inactive.sym,
        stoichiometry={"x": 1},
    )
    deactivation = odax.Reaction(
        name="deactivation",
        rate=k_off.sym * x.sym,
        stoichiometry={"x": -1},
    )

    x_sym, k_on_sym, k_off_sym = sympy.symbols("x k_on k_off")
    return ODETestCase(
        species=[x, inactive],
        parameters=[k_on, k_off],
        reactions=[activation, deactivation],
        expected_derivatives={"x": k_on_sym * (1 - x_sym) - k_off_sym * x_sym},
        expected_vector_field=lambda *, x, k_on, k_off: {
            "x": k_on * (1 - x) - k_off * x
        },
        inputs=[],
        make_args=None,
    )
