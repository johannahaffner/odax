import jax.numpy as jnp
import pytest

from .helpers import ode_systems


@pytest.mark.parametrize("make_system", ode_systems)
@pytest.mark.parametrize("space", ["log", "natural"])
def test_parameter_value_roundtrips(make_system, space) -> None:  # noqa: ANN001
    case = make_system(space=space, trainable=True)
    for param in case.parameters:
        assert jnp.allclose(param.value, jnp.array(param.value))
