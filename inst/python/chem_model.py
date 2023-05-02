import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize

def constant_calc_tri_state(delta_g_df, delta_g_db=None, l=None):

    # Gas constant and temperature
    R, T = 8.314, 298.15

    # Calculate the equilibrium constants
    K_df = jnp.exp(-delta_g_df / (R * T))
    K_db = jnp.exp(-delta_g_db / (R * T))

    # Set arbitrary rate constants and calculate the others based on the equilibrium constants
    k_fp = 1.0
    k_fm = k_fp * K_df
    k_bm = 1.0
    k_b_minus = k_fm * l / K_db
    return k_fp, k_fm, k_bm, k_b_minus

def constant_calc_two_state(delta_g_df):
    # Gas constant and temperature
    R, T = 8.314, 298.15

    # Calculate the equilibrium constants
    K_df = jnp.exp(-delta_g_df / (R * T))
    # Set arbitrary rate constants and calculate the others based on the equilibrium constants
    k_fp = 1.0
    k_fm = k_fp * K_df

    return k_fp, k_fm

# Penalty function
def penalty(x, penalty_weight=1e6):
    constraint_violation = jnp.square(jnp.sum(x) - 1.0)
    return penalty_weight * constraint_violation

# Objective function
def objective_tri_state(x, k_fp, k_fm, k_bm, k_b_minus,l):

    x_o, x_f, x_b = x
    f_xo = -k_fp * x_o + k_fm * x_f
    f_xb = -k_bm * x_b + k_fm * x_f * l
    f_xf = -f_xb - f_xo

    #0 == 1 - x_o - x_f - x_b
    return jnp.square(f_xo) + jnp.square(f_xb) + jnp.square(f_xf) + penalty(x)


def opt_soln_tri_state(delta_g_df, delta_g_db, l=1.0):

    k_fp, k_fm, k_bm, k_b_minus = constant_calc_tri_state(delta_g_df, delta_g_db, l)

    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])

    # Minimize the objective function
    result = minimize(objective_tri_state, x0, method='BFGS', args=(k_fp, k_fm, k_bm, k_b_minus,l))

    print("Optimal solution:", result.x)

    return result.x

def objective_two_state(x,k_fp, k_fm):

    x_o, x_f = x
    f_xo = -k_fp * x_o + k_fm * x_f
    f_xf = 1-f_xo

    return jnp.square(f_xo) + jnp.square(f_xf) + penalty(x)

def opt_soln_two_state(delta_g_df):

    k_fp, k_fm = constant_calc_two_state(delta_g_df)
    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])

    # Minimize the objective function
    result = minimize(objective_two_state, x0, method='BFGS', args=(k_fp, k_fm))

    print("Optimal solution:", result.x)

    return result.x
