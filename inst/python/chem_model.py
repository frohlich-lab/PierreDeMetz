from jaxopt import BFGS
from jaxopt import OptaxSolver
from optax import chain
import jax
import jax.numpy as jnp
from jax import vmap

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
    result = bfgs.run(init_params=x0, delta_g_df=delta_g_df)
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

    result = jnp.square(f_xo) + jnp.square(f_xb) + jnp.square(total_conc) #+ penalty(x)
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


#if __name__ == '__main__':

    test_val = jnp.array([0.12])
    test_val_b = jnp.array([0.4])

    exp = jnp.exp(test_val)
    exp2 = jnp.exp(test_val_b)

    print(opt_2st_vec(test_val))
    print(1/(1+exp))

    print('\n')

    print(opt_3st_vec(test_val, test_val_b))
    print(1/(1+ exp2*(1+exp)))
