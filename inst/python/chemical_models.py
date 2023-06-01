from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap, grad
from diffrax import diffeqsolve, ODETerm, Dopri5
from functools import partial
import equinox as eqx

class ChemicalModel(eqx.Module):
    is_implicit: bool

    def __init__(self, is_implicit):
        self.is_implicit = is_implicit

    def objective_and_grad(self, objective, x, args):
        objective_value = objective(x, args)
        if self.is_implicit:
            grad = jax.grad(objective)(x, args)
        else:
            grad = None  # or calculate gradient differently if needed
        return objective_value, grad

    def solve_implicit(self, x0, args, objective=None):
        if objective is None:
            raise ValueError("An objective function must be provided.")

        #def wrapped_objective_and_grad(x, *args):
            #return self.objective_and_grad(objective, x, args)

        bfgs = BFGS(maxiter=1000, fun=self.objective_and_grad, value_and_grad=True)

        try:
            result = bfgs.run(objective, x0, args)
            print(f'Successfully solved for {result.x}')
            return result.x

        except Exception as e:
            print(f'Error occurred while solving: {e}')
            return None

    def solve_ode(self, objective, x0, args,t0=0, t1=11, dt0=0.1):
        term = ODETerm(objective)
        solver = Dopri5()
        solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=args)
        return solution

    def get_steady_state_solution(self, x0, args, objective):
        solution = self.solve_ode(objective, x0, args)
        steady_state_solution = solution[-1]
        return steady_state_solution

    def opt_vectorize(self, x0, args, objective, t0=None, t1=None, dt0=None):
        if self.is_implicit:
            solve_implicit_vectorized = vmap(self.solve_implicit)
            results = solve_implicit_vectorized(x0, args, objective)
        else:
            solve_ode_vectorized = vmap(self.get_steady_state_solution)
            results = solve_ode_vectorized(objective, x0, args)
        return results

    def solve(self, x0, args, objective=None):
        if self.is_implicit:
            return self.opt_vectorize(x0, args, objective)
        else:
            return self.opt_vectorize(objective, x0, args)


class ThreeStateEquilibrium(ChemicalModel):
    x0_two: jnp.ndarray
    x0_tri: jnp.ndarray

    def __init__(self, is_implicit):
        super().__init__(is_implicit)
        self.x0_two = jnp.array([1/2, 1/2])
        self.x0_tri = jnp.array([1/3, 1/3, 1/3])

    def objective_folding(self, x, args):
        delta_g_df = args
        x_o, x_f = x

        total_conc = 1 - x_o - x_f
        f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
        f_xf = x_o * jnp.exp(-delta_g_df) - x_f

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf]).reshape(-1,)

        else:
            result = jnp.square(f_xo) + jnp.square(f_xf)+jnp.square(total_conc)
            return jnp.squeeze(result)

    def objective_binding(self, x, args):
        delta_g_df, delta_g_db = args
        x_o, x_f, x_b = x
        total_conc = 1 - x_o - x_f - x_b

        f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
        f_xb = x_f * jnp.exp(-delta_g_db) - x_b
        f_xf = -f_xo - f_xb

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf, f_xb]).reshape(-1,)

        else:
            result = jnp.square(f_xo) + jnp.square(f_xf) + jnp.square(f_xb) + jnp.square(total_conc)
            print(result)
            return jnp.squeeze(result)

    def solve_systems(self, args_folding, args_binding):
        if self.is_implicit:
            solution_folding = self.solve(self.x0_two, args=args_folding, objective=self.objective_folding)
            solution_binding = self.solve(self.x0_tri, args=args_binding, objective=self.objective_binding)
        else:
            solution_folding = self.solve(self.x0_two, args=args_folding, objective=self.objective_folding)
            solution_binding = self.solve(self.x0_tri, args=args_binding, objective=self.objective_binding)


        return solution_folding, solution_binding


if __name__ == '__main__':

    three_state_model = ThreeStateEquilibrium(True)

    args_folding = (jnp.array([0.5, 0.2]))
    args_binding = (jnp.array([0.5, 0.2]), jnp.array([0.5, 0.2]))

    solution_folding, solution_binding = three_state_model.solve_systems(args_folding, args_binding)
    print("Solution for folding:", solution_folding)
    print("Solution for binding:", solution_binding)
