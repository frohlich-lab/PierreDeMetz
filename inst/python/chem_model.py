import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize

def constant_calc(delta_g_df, delta_g_db=None, l=None):

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

# Penalty function
def penalty(x, penalty_weight=1e6):
    constraint_violation = jnp.square(jnp.sum(x) - 1.0)
    return penalty_weight * constraint_violation

# Objective function
def objective(x, k_fp, k_fm, k_bm, k_b_minus,l):

    x_o, x_f, x_b = x
    f_xo = -k_fp * x_o + k_fm * x_f
    f_xb = -k_bm * x_b + k_fm * x_f * l
    f_xf = -f_xb - f_xo

    #0 == 1 - x_o - x_f - x_b
    return jnp.square(f_xo) + jnp.square(f_xb) + jnp.square(f_xf) + penalty(x)


def opt_soln(delta_g_df =-10, delta_g_db=-15, l=1.0):

    k_fp, k_fm, k_bm, k_b_minus = constant_calc(delta_g_df, delta_g_db, l)

    # Initial guess
    x0 = jnp.array([1/3, 1/3, 1/3])

    # Minimize the objective function
    result = minimize(objective, x0, method='BFGS', args=(k_fp, k_fm, k_bm, k_b_minus,l))

    print("Optimal solution:", result.x)

    return result.x



if __name__ == "__main__":
    opt_soln(jnp.array([-10,-5]),jnp.array([-15,-12]),jnp.array([1,1]))
