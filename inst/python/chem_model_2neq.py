from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap
from diffrax import diffeqsolve, ODETerm, Dopri5

################# TWO STATE NON EQ MODEL IMPLICIT
def objective_two_state_noneq_implicit(x, delta_g_df):

    pass

def objective_and_grad_two_state_noneq_implicit(x, delta_g_df):
    objective_value = objective_two_state_noneq_implicit(x, delta_g_df)
    grad = jax.grad(objective_two_state_noneq_implicit)(x, delta_g_df)
    return objective_value, grad

def opt_soln_two_state_noneq_implicit(delta_g_df):
    # Initial guess
    x0 = jnp.array([1/2, 1/2])

    # Create a BFGS solver using jaxopt
    bfgs = BFGS(maxiter=100, fun=objective_and_grad_two_state_noneq_implicit, value_and_grad=True)
    result = bfgs.run(init_params=x0,
                      delta_g_df=delta_g_df
                      )
    return result.params[1]

def two_state_noneq_implicit_vec(delta_g_df):
    opt_soln_two_state_vectorized = vmap(opt_soln_two_state_noneq_implicit)
    results = opt_soln_two_state_vectorized(delta_g_df)
    return results

################# TWO STATE NON EQ MODEL ODE
def objective_two_state_noneq_ODE():
    pass

def objective_and_grad_two_state_noneq_ODE():
    pass

def opt_soln_two_state_noneq_ODE():
    pass

def two_state_noneq_ode_vec():
    pass
