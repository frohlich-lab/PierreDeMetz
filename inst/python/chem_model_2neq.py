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
    l = 1.0
    total_conc = 1 - x_f

    f_xf = jnp.exp(-delta_g_df) - x_f

    # OPTIMISATION OBJECTIVE
    result = jnp.square(f_xf) + jnp.square(total_conc)
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
def objective_two_state_noneq_folding_ODE():
    pass

def objective_and_grad_two_state_noneq_folding_ODE():
    pass

def opt_soln_two_state_noneq_folding_ODE():
    pass

def two_state_noneq_folding_ode_vec():
    pass

################# TWO STATE NON EQ MODEL IMPLICIT BINDING
def objective_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db):
    x_b, x_f = x
    l = 1.0
    flux = 1.0

    # TO DO
    f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l
    f_xf = jnp.exp(-delta_g_df) - x_f - f_xb

    # OPTIMISATION OBJECTIVE
    result = jnp.square(f_xb) + jnp.square(f_xf)
    return jnp.squeeze(result)

def objective_and_grad_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db):
    objective_value = objective_two_state_noneq_binding_implicit(x, delta_g_df, delta_g_db)
    grad = jax.grad(objective_two_state_noneq_binding_implicit)(x, delta_g_df, delta_g_db)
    return objective_value, grad

def opt_soln_two_state_noneq_binding_implicit(delta_g_df, delta_g_db):
    # Initial guess
    x0 = jnp.array([1/2, 1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_two_state_noneq_binding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df,
                      delta_g_db=delta_g_db
                      )
    return result.params[0]

def two_state_noneq_binding_implicit_vec(delta_g_df, delta_g_db):
    opt_soln_two_state_vectorized = vmap(opt_soln_two_state_noneq_binding_implicit)
    results = opt_soln_two_state_vectorized(delta_g_df, delta_g_db)
    return results
################# THREE STATE NON EQ MODEL ODE BINDING
def objective_two_state_noneq_binding_ODE():
    pass

def objective_and_grad_two_state_noneq_binding_ODE():
    pass

def opt_soln_two_state_noneq_binding_ODE():
    pass

def two_state_noneq_binding_ode_vec():
    pass
