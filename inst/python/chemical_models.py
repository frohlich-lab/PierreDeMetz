import jax.numpy as jnp
from jax import grad, vmap
from jaxopt import BFGS
from diffrax import ODE, Tsit5

#This file is to have to include every chemical model in one place (apart from explicit)

class StateModel:
    def __init__(self, model='two-state', equilibrium=False, solver='bfgs', maxiter=100, t0=0, t1=10, dt0=0.1):
        self.model = model
        self.equilibrium = equilibrium
        self.solver = solver
        self.maxiter = maxiter
        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0
        self.bfgs = BFGS(maxiter=maxiter, fun=self._objective_and_grad, value_and_grad=True)
        self.ode_solver = Tsit5(ODE(self._ode_model))

    def _compute_fx(self, x, flux, *delta_gs):
        if self.model == 'two-state':
            return [flux * jnp.exp(delta_gs[0]) - x[0],
                    x[0] * jnp.exp(-delta_gs[0]) - x[1]]
        elif self.model == 'three-state':
            l = 1.0
            if self.equilibrium:  # equilibrium model
                return [flux + x[1] - 2 * x[0],
                        -x[2] + x[1] * l - x[2],
                        x[0] - x[1] - (-x[2] + x[1] * l - x[2]) + x[2]]
            else:  # non-equilibrium model
                return [flux * jnp.exp(delta_gs[2]) + x[1] * jnp.exp(delta_gs[0]) - 2 * x[0],
                        -x[2] * jnp.exp(delta_gs[1]) + x[1] * l - x[2] * jnp.exp(delta_gs[3]),
                        x[0] * jnp.exp(-delta_gs[0]) - x[1] - (-x[2] * jnp.exp(delta_gs[1]) + x[1] * l - x[2] * jnp.exp(delta_gs[3])) + x[2] * jnp.exp(delta_gs[3])]

    # ... Rest of the class remains the same

    def solve(self, x0, flux, *delta_gs):
        if self.solver == 'bfgs':
            return self.solve_implicit(x0, flux, *delta_gs)
        elif self.solver == 'ode':
            return self.solve_ode(x0, *delta_gs)

    def solve_vectorized(self, x0, flux, *delta_gs):
        if self.solver == 'bfgs':
            return self.solve_implicit_vectorized(x0, flux, *delta_gs)
        elif self.solver == 'ode':
            return self.solve_ode_vectorized(x0, *delta_gs)
