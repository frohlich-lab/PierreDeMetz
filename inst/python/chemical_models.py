from jaxopt import BFGS
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5
import equinox as eqx

def create_chemical_model(model_type, is_implicit, is_degradation):
    if model_type == 'tri_state_equilibrium':
        return ThreeStateEquilibrium(is_implicit)
    elif model_type == 'two_state_non_equilibrium':
        return TwoStateNonEquilibrium(is_implicit, is_degradation)
    elif model_type == 'tri_state_non_equilibrium':
        return ThreeStateNonEquilibrium(is_implicit)
    else:
        raise ValueError(f'Unknown model_type {model_type}')

class ChemicalModel(eqx.Module):
    is_implicit: bool = eqx.static_field()

    def __init__(self, is_implicit):
        self.is_implicit = is_implicit

    def objective_and_grad(self, x, objective, args):
        objective_value = objective(x, args)
        grad = jax.grad(objective)(x, args)
        return objective_value, grad

    def solve_implicit(self, x0, objective, args):
        bfgs = BFGS(maxiter=1000, fun=self.objective_and_grad, value_and_grad=True)
        try:
            result = bfgs.run(init_params = x0,
                              objective = objective,
                              args = args)
            return result.params
        except Exception as e:
            print(f'Error occurred while solving: {e}')
            return None

    #@jax.jit
    def solve_ode(self, x0, objective, args ,t0=0, t1=10, dt0=0.1):
        term = ODETerm(objective)
        solver = Dopri5()
        solution = diffeqsolve(terms=term, solver=solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=args)
        return solution
    #@jax.jit
    def get_steady_state_solution(self, x0, objective, args):
        solution = self.solve_ode(x0,objective,args)
        steady_state_solution = solution.ys[-1]
        return steady_state_solution
    #@jax.jit
    def opt_vectorize(self, x0, args, objective):
        if self.is_implicit:
            solve_implicit_vectorized = vmap(self.solve_implicit, in_axes=(None, None, 0))
            results = solve_implicit_vectorized(x0, objective, args)
        else:
            solve_ode_vectorized = vmap(self.get_steady_state_solution, in_axes=(None, None, 0))
            results = solve_ode_vectorized(x0, objective, args)
        return results


def flux_folding(x1=1, x2=1, delta_g=None):
    return x1 * jnp.exp(-delta_g) - x2

def flux_binding(x1=1, x2=1, delta_g=None):
    return -x2 * jnp.exp(delta_g) + x1

def flux_degradation(x1=1, delta_g=None):
    return x1 * jnp.exp(delta_g)

def flux_synthesis(flux=1.0,x1=1, delta_g=None):
    return flux * jnp.exp(delta_g) -x1

class ThreeStateEquilibrium(ChemicalModel):
    x0_two: jnp.ndarray #= eqx.static_field()
    x0_tri: jnp.ndarray #= eqx.static_field()

    def __init__(self, is_implicit):
        super().__init__(is_implicit)
        self.x0_two = jnp.array([1/2, 1/2])
        self.x0_tri = jnp.array([1/3, 1/3, 1/3])

    def objective_folding(self, *args):

        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_do, delta_g_df = args

        x_o, x_f = x
        total_conc = 1 - x_o - x_f
        f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
        f_xf = x_o * jnp.exp(-delta_g_df) - x_f

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf]).reshape(-1,)
        else:
            result = jnp.square(f_xo) +jnp.square(f_xf)+ jnp.square(total_conc)
            return jnp.squeeze(result)

    def objective_binding(self, *args):
        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_do, delta_g_df, delta_g_db, delta_g_dd = args

        x_o, x_f, x_b = x
        total_conc = 1 - x_o - x_f - x_b
        #f_xo = - flux_folding(x_o, x_f, delta_g_df)
        #f_xf = flux_folding(x_o, x_f, delta_g_dd) - flux_binding(x_f, x_b, delta_g_db)
        #f_xb = flux_binding(x_f, x_b, delta_g_db)

        f_xo = (-x_o * jnp.exp(-delta_g_df) + x_f)
        f_xb = (- x_b * jnp.exp(delta_g_db) + x_f)
        f_xf =  (x_o * jnp.exp(-delta_g_df) - x_f) - f_xb #(-x_b * jnp.exp(delta_g_db) + x_f )

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf, f_xb]).reshape(-1,)
        else:
            result = jnp.square(f_xo)  + jnp.square(f_xf) + jnp.square(f_xb) + jnp.square(total_conc)
            return jnp.squeeze(result)
    #@jax.jit
    def solve_folding(self, args_folding):
        results = self.opt_vectorize(self.x0_two, args_folding, self.objective_folding)
        return results.T[:][2].reshape(-1, 1)
    #@jax.jit
    def solve_binding(self, args_binding):
        results = self.opt_vectorize(self.x0_tri, args_binding, self.objective_binding)
        return results.T[:][2].reshape(-1, 1)


class TwoStateNonEquilibrium(ChemicalModel):
    x0_two: jnp.ndarray #= eqx.static_field()
    x0_tri: jnp.ndarray #=  eqx.static_field()
    is_degradation: bool = eqx.static_field()

    def __init__(self, is_implicit, is_degradation):
        super().__init__(is_implicit)
        self.x0_two = jnp.array([1/2])
        self.x0_tri = jnp.array([1/2,1/2])
        self.is_degradation = is_degradation

    def objective_folding(self, *args):
        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_do, delta_g_df = args
        x_f = x
        #f_xf = flux_folding(x2=x_f, delta_g =delta_g_df)
        f_xf = jnp.exp(-delta_g_df) - x_f

        if not self.is_implicit:
            return jnp.array([f_xf]).reshape(-1,)
        else:
            result = jnp.square(f_xf)
            return jnp.squeeze(result)

    def objective_binding(self, *args):
        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_do, delta_g_df, delta_g_db, delta_g_dd = args

        x_f, x_b = x
        l = 1.0

        if self.is_degradation==False:
            #f_xb = flux_binding(x1=x_f, x2=x_b, delta_g=delta_g_db)
            #f_xf = flux_folding(x2=x_f, delta_g=delta_g_df) - flux_binding(x1=x_f, x2=x_b, delta_g=delta_g_db)
            f_xf = (jnp.exp(-delta_g_df) - x_f) - (x_b * jnp.exp(delta_g_db) - x_f)
            f_xb = (x_b * jnp.exp(delta_g_db) - x_f)

        else:
            if_xb = flux_binding(x1=x_f, x2=x_b, delta_g=delta_g_db) - flux_degradation(x1=x_b, delta_g=delta_g_dd)
            if_xf = flux_folding(x2=x_f, delta_g=delta_g_df) - flux_binding(x1=x_f, x2=x_b, delta_g=delta_g_db)
            #f_xf = (jnp.exp(-delta_g_df) - x_f) - (x_b * jnp.exp(delta_g_db) - x_f)

        if not self.is_implicit:
            return jnp.array([f_xf, f_xb]).reshape(-1,)
        else:
            result = jnp.square(f_xf) + jnp.square(f_xb)
            return jnp.squeeze(result)

    def solve_folding(self, args_folding):
        results = self.opt_vectorize(self.x0_two, args_folding, self.objective_folding)
        return results.T[:].flatten().reshape(-1, 1)

    def solve_binding(self, args_binding):
        results = self.opt_vectorize(self.x0_tri, args_binding, self.objective_binding)
        return results.T[:][1].reshape(-1, 1)

class ThreeStateNonEquilibrium(ChemicalModel):
    x0_two: jnp.ndarray
    x0_tri: jnp.ndarray
    def __init__(self, is_implicit):
        super().__init__(is_implicit)
        self.x0_two = jnp.array([1/2, 1/2])
        self.x0_tri = jnp.array([1/3, 1/3, 1/3])

    def objective_folding(self, *args):
        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_do, delta_g_df = args

        x_o, x_f = x
        flux = 1.0

        #f_xo = flux_synthesis(flux, x_o, delta_g_do) - flux_folding(x_o, x_f, delta_g_df)
        #f_xf = flux_folding(x_o, x_f, delta_g_df)

        f_xo = - (x_o * jnp.exp(-delta_g_df) - x_f) + (jnp.exp(-delta_g_do) - x_o)
        f_xf = (x_o * jnp.exp(-delta_g_df) - x_f)

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf]).reshape(-1,)
        else:
            result = jnp.square(f_xo) + jnp.square(f_xf)
            return jnp.squeeze(result)

    def objective_binding(self, *args):
        if not self.is_implicit:
            t, x, args = args
        else:
            x, args = args

        delta_g_df, delta_g_db,delta_g_dd, delta_g_do = args
        x_o, x_f, x_b = x

        flux = 1.0
        l = 1.0

        #f_xo = flux_synthesis(flux, x_o, delta_g_do) - flux_folding(x_o, x_f, delta_g_df)
        #f_xb = flux_binding(x_f, x_b, delta_g_db) - flux_degradation(x_b, delta_g_dd)
        #f_xf = flux_folding(x_o, x_f, delta_g_df) - flux_binding(x_f, x_b, delta_g_db)

        f_xo = - (x_o * jnp.exp(-delta_g_df) - x_f) + (jnp.exp(delta_g_do) - x_o)
        f_xf = (x_o * jnp.exp(-delta_g_df) - x_f) - (x_b * jnp.exp(delta_g_db) - x_f)
        f_xb = (x_b * jnp.exp(delta_g_db) - x_f) - x_b * jnp.exp(-delta_g_dd)

        if not self.is_implicit:
            return jnp.array([f_xo, f_xf, f_xb]).reshape(-1,)
        else:
            result = jnp.square(f_xo) + jnp.square(f_xf) + jnp.square(f_xb)
            return jnp.squeeze(result)

    def solve_folding(self, args_folding):
        results = self.opt_vectorize(self.x0_two, args_folding, self.objective_folding)
        return results.T[:][2].reshape(-1, 1)

    def solve_binding(self, args_binding):
        results = self.opt_vectorize(self.x0_tri, args_binding, self.objective_binding)
        return results.T[:][2].reshape(-1, 1)

if __name__ == '__main__':
    delta_g_df = jnp.array([0.5, 0.2])
    delta_g_db = jnp.array([0.5, 0.2])
    delta_g_dd = jnp.array([0.5,0.3])
    delta_g_do = jnp.array([0.5,0.3])

    args_folding = (delta_g_do, delta_g_df)
    args_binding = (delta_g_do, delta_g_df, delta_g_db, delta_g_dd)

    ###THRE STATE EQUILIBRIUM
    three_state_model = ThreeStateEquilibrium(is_implicit = True)


    solution_folding = three_state_model.solve_folding(args_folding)
    solution_binding = three_state_model.solve_binding(args_binding)
    print('Three State Equilibrium')
    print("Solution for folding:", solution_folding)
    print("Solution for binding:", solution_binding)

    ### TWO STATE NON EQUILIBRIUM
    two_state_non_eq_model = TwoStateNonEquilibrium(is_implicit = True,
                                              is_degradation = False)


    solution_folding = two_state_non_eq_model.solve_folding(args_folding)
    solution_binding = two_state_non_eq_model.solve_binding(args_binding)

    print('Two State Non Equilibrium')
    print("Solution for folding:", solution_folding)
    print("Solution for binding:", solution_binding)

    ### THREE STATE NON EQUILIBRIUM
    three_state_non_eq_model = ThreeStateNonEquilibrium(is_implicit = True)

    solution_folding = three_state_non_eq_model.solve_folding(args_folding)
    solution_binding = three_state_non_eq_model.solve_binding(args_binding)

    print('Three State Non Equilibrium')
    print("Solution for folding:", solution_folding)
    print("Solution for binding:", solution_binding)
