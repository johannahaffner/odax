import pytest
import sympy

from .helpers import ode_systems


@pytest.mark.parametrize("make_system", ode_systems)
def test_species_sym(make_system) -> None:  # noqa: ANN001
    case = make_system(space="log", trainable=True)
    for species in case.species:
        assert species.sym == sympy.Symbol(species.name)
