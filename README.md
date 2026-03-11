# odax

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