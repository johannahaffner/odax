# Odax

Parsimoniously express your differential equation models with parameter constraints + forcing terms in JAX.
This is a wrapper around [sympy2jax](https://github.com/patrick-kidger/sympy2jax) that adds support for

- parameter constraints via [Paramax](https://github.com/danielward27/paramax)
- forcing terms or experimental conditions $u(t)$, e.g. expressed via [Diffrax](https://github.com/patrick-kidger/diffrax)
- lots of different variations of reaction equations - built from the same species + parameters, endlessly tweakable with minimal duplicated code.

Odax implements a subset of SBML specifications - expressing models via species, parameters, differential and algebraic equations, and arbitrary forcing terms.
It is tightly scoped to this bit only - and comes in at just under 250 lines of code + tests.

## Installation

```bash
pip install odax
```
Requires Python 3.11+. 

## Quick example

```python
import jax.numpy as jnp

from odax import Model, Parameter, Reaction, Species

x = Species(name="x")
alpha = Parameter(name="alpha", value=1.0, trainable=True, space="log")
k = Parameter(name="k", value=0.5, trainable=True, space="log")

model = Model(
    species=[x],
    parameters=[alpha, k],
    reactions=[
        Reaction(name="production", rate=alpha.sym, stoichiometry={"x": 1}),
        Reaction(name="decay", rate=k.sym * x.sym, stoichiometry={"x": -1}),
    ],
)

dydt = model(t=jnp.array(0.0), y={"x": jnp.array(1.0)}, args=None)
# {"x": Array(0.5)} — i.e. alpha - k*x = 1.0 - 0.5*1.0
```

## See also: other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.   

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): nonlinear solvers.
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.   
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  