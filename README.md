# odax

Parsimoniously express your ODE models with parameter constraints + forcing terms in JAX.
This is a wrapper around [sympy2jax](https://github.com/patrick-kidger/sympy2jax) that adds support for

- parameter constraints via [Paramax](https://github.com/danielward27/paramax)
- forcing terms or experimental conditions $u(t)$, e.g. expressed via [Diffrax](https://github.com/patrick-kidger/diffrax)
- lots of different variations of reaction equations - these are essentially swappable model components, no need to duplicate code!

In particular, those needs frequently arise in systems biology / quantitative systems pharmacometry / PKPD modelling contexts. In the parlance of the latter, odax expresses structural models, in the parlance of everyone else: it gives you an expression for the right-hand side of the vector field.

Odax implements a subset of SBML specifications - expressing models via species, parameters, differential and algebraic equations, and arbitrary forcing terms.
It is tightly scoped to this bit only - and comes in at just under 250 lines of code.

## Installation

```bash
pip install odax
```
Requires Python 3.11+. 