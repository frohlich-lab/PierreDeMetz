from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap, grad
from diffrax import diffeqsolve, ODETerm, Dopri5
from functools import partial

class ChemicalModel:
    def __init__(self, solver):
        self.solver = solver

    def objective_and_grad(self, objective, x, args):
        objective_value = objective(x, args)
        grad = jax.grad(objective)(x, args)
        return objective_value, grad

    def solve_implicit(self, x0, args, objective=None):
        if objective is None:
            raise ValueError("An objective function must be provided.")

        bfgs = BFGS(maxiter=1000, fun=self.objective_and_grad, value_and_grad=True)

        try:
            result = bfgs.run(init_params=x0, objective=objective, x=x0, args=args)
            print(f'Successfully solved for {result.x}')
            return result.x

        except Exception as e:
            print(f'Error occurred while solving: {e}')
            return None

    def solve_ode(self, objective, t0, t1, dt0, x0, args):
        term = ODETerm(objective)
        solver = Dopri5()
        solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=args)
        return solution

    def opt_vectorize(self, x, args):
        opt_soln_tri_state_vectorized = vmap(self.solve_implicit)
        results = opt_soln_tri_state_vectorized(args)
        return results

class ThreeStateEquilibrium(ChemicalModel):
    def __init__(self, solver):
        super().__init__(solver)
        self.x0_two = jnp.array([1/2, 1/2])
        self.x0_tri = jnp.array([1/3, 1/3, 1/3])


    def objective_folding(self, x, args):
        delta_g_df, = args
        x_o, x_f = x
        total_conc = 1 - x_o - x_f
        f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
        f_xf = x_o * jnp.exp(-delta_g_df) - x_f

        if self.solver == 'ODE':
            return jnp.array([f_xo, f_xf]).reshape(-1,)

        elif self.solver == 'implicit':
            return jnp.squeeze(jnp.square(f_xo) + jnp.square(f_xf)+jnp.square(total_conc))

    def objective_binding(self, x, args):
        l, delta_g_df, delta_g_db = args
        x_o, x_f, x_b = x
        total_conc = 1 - x_o - x_f - x_b

        f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
        f_xb = x_f * jnp.exp(-delta_g_db) - x_b
        f_xf = -f_xo - f_xb

        if self.solver == 'ODE':
            return jnp.array([f_xo, f_xf, f_xb]).reshape(-1,)

        elif self.solver == 'implicit':
            result = jnp.square(f_xo) + jnp.square(f_xf) + jnp.square(f_xb) + jnp.square(total_conc)
            return jnp.squeeze(result)

    def get_objective_folding_wrapper(self):
        def wrapper(x, args):
            return self.objective_folding(x, args)
        return wrapper

    def get_objective_binding_wrapper(self):
        def wrapper(x, args):
            return self.objective_binding(x, args)
        return wrapper

    def solve_systems(self, args_folding, args_binding):
        folding_wrapper = self.get_objective_folding_wrapper()
        binding_wrapper = self.get_objective_binding_wrapper()

        solution_folding = self.solve_implicit(self.x0_two, args=args_folding, objective=folding_wrapper)
        solution_binding = self.solve_implicit(self.x0_two, args=args_binding, objective=binding_wrapper)

        return solution_folding, solution_binding

if __name__ == '__main__':
    # Instantiate the three-state equilibrium model
    three_state_model = ThreeStateEquilibrium('implicit')

    # Mock values
    args_folding = (0.5,)  # delta_g_df
    args_binding = (1.0, 0.5, 0.2)  # l, delta_g_df, delta_g_db

    # Solve the systems and print the solutions
    solution_folding, solution_binding = three_state_model.solve_systems(args_folding, args_binding)
    print("Solution for folding:", solution_folding)
    print("Solution for binding:", solution_binding)
