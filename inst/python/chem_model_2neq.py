from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5

################# TWO STATE NON EQ MODEL IMPLICIT FOLDING
def objective_two_state_noneq_folding_implicit(x, delta_g_df):
    x_f = x
    f_xf = jnp.exp(-delta_g_df) - x_f

    # OPTIMISATION OBJECTIVE
    result = jnp.square(f_xf) #+ jnp.square(total_conc)
    return jnp.squeeze(result)

def objective_and_grad_two_state_noneq_folding_implicit(x, delta_g_df):
    objective_value = objective_two_state_noneq_folding_implicit(x, delta_g_df)
    grad = jax.grad(objective_two_state_noneq_folding_implicit)(x, delta_g_df)
    return objective_value, grad

def opt_soln_two_state_noneq_folding_implicit(delta_g_df):
    # Initial guess
    x0 = jnp.array([1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_two_state_noneq_folding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df
                      )
    return result.params[0]

def two_state_noneq_folding_implicit_vec(delta_g_df):
    opt_soln_two_state_vectorized = vmap(opt_soln_two_state_noneq_folding_implicit)
    results = opt_soln_two_state_vectorized(delta_g_df)
    return results

################# TWO STATE NON EQ MODEL ODE FOLDING
def objective_two_state_noneq_folding_ODE(t, x, args):
    delta_g_df = args
    x_f = x
    dx_f_dt = jnp.exp(-delta_g_df) - x_f
    return jnp.array([dx_f_dt]).reshape(-1,)

def objective_and_grad_two_state_noneq_folding_ODE(delta_g_df, x0, t0=0, t1=10, dt0=0.1):
    term = ODETerm(objective_two_state_noneq_folding_ODE)
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(delta_g_df))
    return solution

def opt_soln_two_state_noneq_folding_ODE(delta_g_df):
    x0 = jnp.array([1/2])
    solution = objective_and_grad_two_state_noneq_folding_ODE(delta_g_df, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[0]])

def two_state_noneq_folding_ode_vec(delta_g_df):
    ss_two_state_vectorized = vmap(opt_soln_two_state_noneq_folding_ODE)
    results = ss_two_state_vectorized(delta_g_df=delta_g_df)
    return results.flatten()

################# TWO STATE NON EQ MODEL IMPLICIT BINDING
def objective_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_dd):
    x_b, x_f = x
    l = 1.0
    #old version
    #f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l
    #f_xf = jnp.exp(-delta_g_df) - x_f - f_xb

    #new version w/ degradation
    f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l - x_b * jnp.exp(delta_g_dd)
    f_xf = jnp.exp(-delta_g_df) - x_f - f_xb + x_b * jnp.exp(delta_g_dd)

    result = jnp.square(f_xb) + jnp.square(f_xf)
    return jnp.squeeze(result)

def objective_and_grad_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_dd):
    objective_value = objective_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_dd)
    grad = jax.grad(objective_two_state_noneq_binding_implicit)(x, delta_g_df, delta_g_db, delta_g_dd)
    return objective_value, grad

def opt_soln_two_state_noneq_binding_implicit(delta_g_df, delta_g_db, delta_g_dd):
    # Initial guess
    x0 = jnp.array([1/2, 1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_two_state_noneq_binding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df,
                      delta_g_db=delta_g_db,
                      delta_g_dd=delta_g_dd
                      )
    return result.params[0]

def two_state_noneq_binding_implicit_vec(delta_g_df, delta_g_db, delta_g_dd):
    opt_soln_two_state_vectorized = vmap(opt_soln_two_state_noneq_binding_implicit)
    results = opt_soln_two_state_vectorized(delta_g_df, delta_g_db, delta_g_dd)
    return results
################# TWO STATE NON EQ MODEL ODE BINDING
def objective_two_state_noneq_binding_ODE(t, x, args):
    l, delta_g_df, delta_g_db, delta_g_dd = args
    x_f, x_b = x
    dx_b_dt = -x_b * jnp.exp(delta_g_db) + x_f * l - x_b * jnp.exp(delta_g_dd)
    dx_f_dt = jnp.exp(-delta_g_df) - x_f - dx_b_dt + x_b * jnp.exp(delta_g_dd)

    return jnp.array([dx_f_dt, dx_b_dt]).reshape(-1,)

def objective_and_grad_two_state_noneq_binding_ODE(l, delta_g_df, delta_g_db, delta_g_dd, x0, t0=0, t1=10, dt0=0.1):
    term = ODETerm(objective_two_state_noneq_binding_ODE)
    solver = Dopri5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(l, delta_g_df, delta_g_db,  delta_g_dd))
    return solution

def opt_soln_two_state_noneq_binding_ODE(l,delta_g_df, delta_g_db, delta_g_dd):
    x0 = jnp.array([1/2, 1/2])
    solution = objective_and_grad_two_state_noneq_binding_ODE(l, delta_g_df, delta_g_db, delta_g_dd, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[1]])

def two_state_noneq_binding_ode_vec(delta_g_df, delta_g_db, delta_g_dd):
    l_val = 1.0
    l = jnp.repeat(l_val, repeats=delta_g_df.shape[0])
    ss_tri_state_vectorized = vmap(opt_soln_two_state_noneq_binding_ODE)
    results = ss_tri_state_vectorized(l=l, delta_g_df=delta_g_df,
                                      delta_g_db=delta_g_db,
                                      delta_g_dd = delta_g_dd)
    return results.flatten()
