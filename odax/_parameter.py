from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import paramax as px
import sympy
from jaxtyping import Array


class Parameter(eqx.Module):
    name: str
    trainable: bool
    space: Literal["log", "natural"]
    param: px.AbstractUnwrappable

    def __init__(
        self,
        name: str,
        value: float,
        *,
        trainable: bool,
        space: Literal["log", "natural"],
    ) -> None:
        self.name = name
        self.trainable = trainable
        self.space = space

        v: px.AbstractUnwrappable
        if space == "log":
            v = px.Parameterize(jnp.exp, jnp.log(jnp.array(value)))
        elif space == "natural":
            v = px.Parameterize(lambda x: x, jnp.array(value))
        if not trainable:
            v = px.NonTrainable(v)
        self.param = v

    @property
    def sym(self) -> sympy.Symbol:
        return sympy.Symbol(self.name)

    @property
    def value(self) -> Array:
        return px.unwrap(self.param)
