from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5

################# TWO STATE MODEL
def objective_two_state(x, delta_g_df):
    x_o, x_f = x
    total_conc = 1 - x_o - x_f
    f_xo = - jnp.exp(-delta_g_df)* x_o + x_f
    result = jnp.square(f_xo) + jnp.square(total_conc)
    return jnp.squeeze(result)

def objective_and_grad_two_state(x, delta_g_df):
    objective_value = objective_two_state(x, delta_g_df)
    grad = jax.grad(objective_two_state)(x, delta_g_df)
    return objective_value, grad

def opt_soln_two_state(delta_g_df):

    # Initial guess
    x0 = jnp.array([1/2, 1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_two_state, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df
                      )
    return result.params[1]


def opt_2st_vec(delta_g_df):
    opt_soln_two_state_vectorized = vmap(opt_soln_two_state)
    results = opt_soln_two_state_vectorized(delta_g_df)
    return results

################ THREE STATE MODEL
def objective_tri_state(x, l, delta_g_df, delta_g_db):
    x_o, x_f, x_b = x
    total_conc = 1 - x_o - x_b - x_f

    f_xo = -x_o * jnp.exp(-delta_g_df) + x_f
    f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l

    result = jnp.square(f_xo) + jnp.square(f_xb) + jnp.square(total_conc)
    return jnp.squeeze(result)

def objective_and_grad_tri_state(x, l, delta_g_df, delta_g_db):
    objective_value = objective_tri_state(x, l, delta_g_df, delta_g_db)
    grad = jax.grad(objective_tri_state)(x, l, delta_g_df, delta_g_db)
    return objective_value, grad

def opt_soln_tri_state(l, delta_g_df, delta_g_db):

    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])
    bfgs = BFGS(maxiter=1000, fun=objective_and_grad_tri_state, value_and_grad=True)

    # Minimize the objective function

    result = bfgs.run(init_params=x0,
                      l=l,
                      delta_g_df=delta_g_df,
                      delta_g_db=delta_g_db
                      )

    # Return the optimal solution
    return result.params[2]

def opt_3st_vec(delta_g_df, delta_g_db, l_val=1.0):
    l = jnp.repeat(l_val, repeats=delta_g_df.shape[0])
    opt_soln_tri_state_vectorized = vmap(opt_soln_tri_state)
    results = opt_soln_tri_state_vectorized(l, delta_g_df=delta_g_df, delta_g_db=delta_g_db)
    return results


################ TWO STATE ODE MODEL
def dx_dt_two_state(t, x, args):
    delta_g_df, = args
    x_o, x_f = x
    dx_o_dt = -x_o * jnp.exp(-delta_g_df) + x_f
    dx_f_dt = x_o * jnp.exp(-delta_g_df) - x_f
    return jnp.array([dx_o_dt, dx_f_dt]).reshape(-1,)

def solve_two_state_ode(delta_g_df, x0, t0=0, t1=10, dt0=0.1):
    term = ODETerm(dx_dt_two_state)
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(delta_g_df,))
    return solution

def get_steady_state_solution_two_state(delta_g_df):
    x0 = jnp.array([1/2,1/2])
    solution = solve_two_state_ode(delta_g_df, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[1]])

def ss_two_state_vec(delta_g_df):
    ss_two_state_vectorized = vmap(get_steady_state_solution_two_state)
    results = ss_two_state_vectorized(delta_g_df=delta_g_df)
    return results.flatten()

################ THREE STATE ODE MODEL
def dx_dt_tri_state(t, x, args):
    l, delta_g_df, delta_g_db = args
    x_o, x_f, x_b = x
    dx_o_dt = -x_o * jnp.exp(-delta_g_df) + x_f
    dx_b_dt = x_f * jnp.exp(-delta_g_db) - x_b
    dx_f_dt = -dx_o_dt - dx_b_dt
    return jnp.array([dx_o_dt, dx_f_dt, dx_b_dt]).reshape(-1,)

def solve_tri_state_ode(l, delta_g_df, delta_g_db, x0, t0=0, t1=10, dt0=0.1):
    term = ODETerm(dx_dt_tri_state)
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(l, delta_g_df, delta_g_db))
    return solution

# Extract the steady-state solution
def get_steady_state_solution_tri_state(l,delta_g_df, delta_g_db):
    x0 = jnp.array([1/3, 1/3, 1/3])
    solution = solve_tri_state_ode(l, delta_g_df, delta_g_db, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[2]])

def ss_tri_state_vec(delta_g_df, delta_g_db):
    l_val = 1.0
    l = jnp.repeat(l_val, repeats=delta_g_df.shape[0])
    ss_tri_state_vectorized = vmap(get_steady_state_solution_tri_state)
    results = ss_tri_state_vectorized(l=l, delta_g_df=delta_g_df, delta_g_db=delta_g_db)
    return results.flatten()

if __name__ == '__main__':
    delta_g_df = jnp.array([1/2])
    delta_g_db = jnp.array([1/2])
    print(ss_two_state_vec(delta_g_df))
    print(ss_tri_state_vec(delta_g_df, delta_g_db))
