from typing import Protocol, runtime_checkable

import equinox as eqx
import sympy
from jaxtyping import Array, Float


@runtime_checkable
class AbstractPath(Protocol):
    """Scalar-valued path evaluated at a given time.

    Each input is intentionally scalar. Vector-valued signals must be split into
    separate Input objects per component — one named symbol maps to one scalar value
    throughout the sympy assembly.

    diffrax.LinearInterpolation satisfies this protocol when ys has shape (T,).
    """

    def evaluate(self, t: Float[Array, ""]) -> Float[Array, ""]: ...


class Input(eqx.Module):
    name: str

    @property
    def sym(self) -> sympy.Symbol:
        return sympy.Symbol(self.name)
