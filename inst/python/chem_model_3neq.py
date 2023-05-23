from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5

################# THREE STATE NON EQ MODEL IMPLICIT FOLDING
def objective_three_state_noneq_folding_implicit(x, delta_g_df, delta_g_do):
    x_o, x_f = x
    l = 1.0
    flux = 1.0

    f_xo = -x_f * jnp.exp(delta_g_df) + flux * delta_g_do
    f_xf = jnp.exp(-delta_g_df) - x_f

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
def objective_three_state_noneq_folding_ODE():
    pass

def objective_and_grad_three_state_noneq_folding_ODE():
    pass

def opt_soln_three_state_noneq_folding_ODE():
    pass

def three_state_noneq_folding_ode_vec():
    pass

################# THREE STATE NON EQ MODEL IMPLICIT BINDING
def objective_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do):
    x_o, x_f, x_b = x
    l = 1.0
    flux = 1.0

    # TO DO
    f_xo = -x_f * jnp.exp(delta_g_df) + flux * delta_g_do
    f_xb = -x_b * jnp.exp(delta_g_db) + x_f * l
    f_xf = jnp.exp(-delta_g_df) - x_f - f_xb

    # OPTIMISATION OBJECTIVE
    result = jnp.square(f_xb) + jnp.square(f_xf) + jnp.square(f_xo)
    return jnp.squeeze(result)

def objective_and_grad_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do):
    objective_value = objective_three_state_noneq_binding_implicit(x, delta_g_df, delta_g_db, delta_g_do)
    grad = jax.grad(objective_three_state_noneq_binding_implicit)(x, delta_g_df, delta_g_db, delta_g_do)
    return objective_value, grad

def opt_soln_three_state_noneq_binding_implicit(delta_g_df, delta_g_db, delta_g_do):
    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_three_state_noneq_binding_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df,
                      delta_g_db=delta_g_db,
                      delta_g_do=delta_g_do
                      )
    return result.params[2]

def three_state_noneq_binding_implicit_vec(delta_g_df, delta_g_db, delta_g_do):
    opt_soln_three_state_vectorized = vmap(opt_soln_three_state_noneq_binding_implicit)
    results = opt_soln_three_state_vectorized(delta_g_df, delta_g_db, delta_g_do)
    return results

################# THREE STATE NON EQ MODEL ODE BINDING
def objective_three_state_noneq_binding_ODE():
    pass

def objective_and_grad_three_state_noneq_binding_ODE():
    pass

def opt_soln_three_state_noneq_binding_ODE():
    pass

def three_state_noneq_binding_ode_vec():
    pass
