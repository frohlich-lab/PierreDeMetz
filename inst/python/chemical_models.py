import jax.numpy as jnp
from jax import grad, vmap
from jaxopt import BFGS
from diffrax import ODE, Tsit5

#This file is to have to include every chemical model in one place (apart from explicit)

class StateModel:
    def __init__(self, equilibrium=False, maxiter=100, t0=0, t1=10, dt0=0.1):
        self.equilibrium = equilibrium
        self.maxiter = maxiter
        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0
        self.bfgs = BFGS(maxiter=maxiter, fun=self._objective_and_grad, value_and_grad=True)
        self.ode_solver = Tsit5(ODE(self._ode_model))

    def _compute_fx(self, x, flux, *delta_gs):
        if len(x) == 2: # two-state model
            return [flux * jnp.exp(delta_gs[0]) - x[0],
                    x[0] * jnp.exp(-delta_gs[0]) - x[1]]

        elif len(x) == 3: # three-state model
            l = 1.0
            if self.equilibrium:  # equilibrium model
                return [flux + x[1] - 2 * x[0],
                        -x[2] + x[1] * l - x[2],
                        x[0] - x[1] - (-x[2] + x[1] * l - x[2]) + x[2]]
            else:  # non-equilibrium model
                return [flux * jnp.exp(delta_gs[2]) + x[1] * jnp.exp(delta_gs[0]) - 2 * x[0],
                        -x[2] * jnp.exp(delta_gs[1]) + x[1] * l - x[2] * jnp.exp(delta_gs[3]),
                        x[0] * jnp.exp(-delta_gs[0]) - x[1] - (-x[2] * jnp.exp(delta_gs[1]) + x[1] * l - x[2] * jnp.exp(delta_gs[3])) + x[2] * jnp.exp(delta_gs[3])]


    def _objective(self, x, flux, *delta_gs):
        f_x = self._compute_fx(x, flux, *delta_gs)
        return jnp.sum(jnp.square(f_x))

    def _objective_and_grad(self, x, flux, *delta_gs):
        objective_value = self._objective(x, flux, *delta_gs)
        grad_value = grad(self._objective)(x, flux, *delta_gs)
        return objective_value, grad_value

    def _ode_model(self, x, t, args):
        flux = 1.0
        return jnp.array(self._compute_fx(x, flux, *args)).reshape(-1,)

    def solve_implicit(self, x0, flux, *delta_gs):
        result = self.bfgs.run(init_params=x0, flux=flux, *delta_gs)
        return result.params

    def solve_ode(self, x0, *delta_gs):
        solution = self.ode_solver(init_state=x0, init_time=self.t0, final_time=self.t1, delta_g_vals=delta_gs)
        return solution.state[-1]

    def solve_implicit_vectorized(self, x0, flux, *delta_gs):
        vectorized_solver = vmap(self.solve_implicit)
        return vectorized_solver(x0, flux, *delta_gs)

    def solve_ode_vectorized(self, x0, *delta_gs):
        vectorized_solver = vmap(self.solve_ode)
        return vectorized_solver(x0, *delta_gs)
