from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5

################# THREE STATE NON EQ MODEL IMPLICIT FOLDING
def objective_three_state_noneq_folding_implicit(x, delta_g_df, delta_g_do):
    x_o, x_f = x
    flux = 1.0

    #new version
    f_xo = flux * jnp.exp(delta_g_do) + x_f * jnp.exp(delta_g_df) -2 * x_o
    f_xf = x_o * jnp.exp(-delta_g_df) - x_f

    # OPTIMISATION OBJECTIVE
    result = jnp.square(f_xf) +  jnp.square(f_xo)
    return jnp.squeeze(result)

def objective_and_grad_three_state_noneq_folding_implicit(x, delta_g_df, delta_g_do):
    objective_value = objective_three_state_noneq_folding_implicit(x, delta_g_df, delta_g_do)
    grad = jax.grad(objective_three_state_noneq_folding_implicit)(x, delta_g_df, delta_g_do)
    return objective_value, grad

def opt_soln_three_state_noneq_folding_implicit(delta_g_df, delta_g_do):
    # Initial guess
    x0 = jnp.array([1/2, 1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_three_state_noneq_folding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df,
                      delta_g_do=delta_g_do
                      )
    return result.params[1]

def three_state_noneq_folding_implicit_vec(delta_g_df, delta_g_do):
    opt_soln_three_state_vectorized = vmap(opt_soln_three_state_noneq_folding_implicit)
    results = opt_soln_three_state_vectorized(delta_g_df, delta_g_do)
    return results

################# THREE STATE NON EQ MODEL ODE FOLDING
def objective_three_state_noneq_folding_ODE(t, x, args):
    delta_g_df, delta_g_do = args
    flux=1.0
    x_f, x_o = x
    dx_o_dt = flux * jnp.exp(delta_g_do) + x_f * jnp.exp(delta_g_df) -2 * x_o
    dx_f_dt = x_o * jnp.exp(-delta_g_df) - x_f
    return jnp.array([dx_o_dt, dx_f_dt]).reshape(-1,)

def objective_and_grad_three_state_noneq_folding_ODE(delta_g_df, delta_g_do, x0, t0=0, t1=10, dt0=0.1):
    term = ODETerm(objective_three_state_noneq_folding_ODE)
    solver = Tsit5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(delta_g_df,delta_g_do))
    return solution

def opt_soln_three_state_noneq_folding_ODE(delta_g_df,delta_g_do):
    x0 = jnp.array([1/2,1/2])
    solution = objective_and_grad_three_state_noneq_folding_ODE(delta_g_df,delta_g_do, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[1]])

def three_state_noneq_folding_ode_vec(delta_g_df, delta_g_do):
    ss_two_state_vectorized = vmap(opt_soln_three_state_noneq_folding_ODE)
    results = ss_two_state_vectorized(delta_g_df=delta_g_df, delta_g_do=delta_g_do)
    return results.flatten()

################# THREE STATE NON EQ MODEL IMPLICIT BINDING
def objective_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do, delta_g_dd):
    x_o, x_f, x_b = x
    l = 1.0
    flux = 1.0

    f_xo = flux * jnp.exp(delta_g_do) + x_f * jnp.exp(delta_g_df) -2 * x_o
    f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l - x_b * jnp.exp(delta_g_dd)
    f_xf = x_o * jnp.exp(-delta_g_df) - x_f - f_xb + - x_b * jnp.exp(delta_g_dd)

    result = jnp.square(f_xb) + jnp.square(f_xf) + jnp.square(f_xo)
    return jnp.squeeze(result)

def objective_and_grad_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do, delta_g_dd):
    objective_value = objective_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do, delta_g_dd)
    grad = jax.grad(objective_three_state_noneq_binding_implicit)(x, delta_g_df, delta_g_db, delta_g_do, delta_g_dd)
    return objective_value, grad

def opt_soln_three_state_noneq_binding_implicit(delta_g_df, delta_g_db, delta_g_do, delta_g_dd):
    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_three_state_noneq_binding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df,
                      delta_g_db=delta_g_db,
                      delta_g_do=delta_g_do,
                      delta_g_dd=delta_g_dd
                      )
    return result.params[2]

def three_state_noneq_binding_implicit_vec(delta_g_df, delta_g_db, delta_g_do, delta_g_dd):
    opt_soln_three_state_vectorized = vmap(opt_soln_three_state_noneq_binding_implicit)
    results = opt_soln_three_state_vectorized(delta_g_df, delta_g_db, delta_g_do, delta_g_dd)
    return results

################# THREE STATE NON EQ MODEL ODE BINDING
def objective_three_state_noneq_binding_ODE(t,x,args):
    l, delta_g_df,delta_g_do, delta_g_db, delta_g_dd = args
    x_o, x_f, x_b  = x
    flux = 1.0

    dx_o_dt = flux * jnp.exp(delta_g_do) + x_f * jnp.exp(delta_g_df) -2 * x_o
    dx_b_dt = -x_b * jnp.exp(delta_g_db) + x_f * l - x_b * jnp.exp(delta_g_dd)
    dx_f_dt = x_o * jnp.exp(-delta_g_df) - x_f - dx_b_dt + x_b * jnp.exp(delta_g_dd)

    return jnp.array([dx_o_dt, dx_f_dt, dx_b_dt]).reshape(-1,)

def objective_and_grad_three_state_noneq_binding_ODE(l, delta_g_df, delta_g_do, delta_g_db, delta_g_dd, x0, t0=0, t1=10, dt0=0.01):
    term = ODETerm(objective_three_state_noneq_binding_ODE)
    solver = Tsit5()
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=x0, args=(l, delta_g_df,delta_g_do, delta_g_db, delta_g_dd))
    return solution

def opt_soln_three_state_noneq_binding_ODE(l,delta_g_df, delta_g_do, delta_g_db, delta_g_dd):
    x0 = jnp.array([1/10,1/10,1/10])
    solution = objective_and_grad_three_state_noneq_binding_ODE(l, delta_g_df, delta_g_do, delta_g_db, delta_g_dd, x0)
    steady_state_solution = solution.ys[-1]
    return jnp.array([steady_state_solution[2]])

def three_state_noneq_binding_ode_vec(delta_g_db, delta_g_df, delta_g_do, delta_g_dd):
    l_val = 1.0
    l = jnp.repeat(l_val, repeats=delta_g_df.shape[0])
    ss_tri_state_vectorized = vmap(opt_soln_three_state_noneq_binding_ODE)
    results = ss_tri_state_vectorized(l=l, delta_g_df=delta_g_df,
                                      delta_g_do= delta_g_do,
                                      delta_g_db=delta_g_db,
                                      delta_g_dd=delta_g_dd)
    return results.flatten()



if __name__ == '__main__':
    delta_g_df = jnp.array([0.5, 0.2])
    delta_g_db = jnp.array([0.5, 0.2])
    delta_g_dd = jnp.array([0.5,0.3])
    delta_g_do = jnp.array([0.5,0.3])

    print(three_state_noneq_folding_implicit_vec(delta_g_df,delta_g_do))
    print(three_state_noneq_binding_implicit_vec(delta_g_df, delta_g_do, delta_g_db, delta_g_dd))
