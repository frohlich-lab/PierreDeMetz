import jax.numpy as jnp
import haiku as hk
from chem_model_3eq import opt_2st_vec, opt_3st_vec, ss_two_state_vec, ss_tri_state_vec
from chem_model_2neq import two_state_noneq_folding_implicit_vec, two_state_noneq_binding_implicit_vec, two_state_noneq_folding_ode_vec, two_state_noneq_binding_ode_vec
from chem_model_3neq import three_state_noneq_binding_implicit_vec, three_state_noneq_folding_implicit_vec, three_state_noneq_binding_ode_vec, three_state_noneq_folding_ode_vec

##################### IMPLEMENTATION OF THE THREE MODELS #####################
class StateProbFolded(hk.Module):
    def __init__(self, model_type='tri_state_equilibrium_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, binding, folding, synthesis=None, degradation=None):
        if self.model_type == 'tri_state_equilibrium_explicit':
            return 1/(1+jnp.exp(folding))
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return opt_2st_vec(folding).reshape(-1, 1)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return ss_two_state_vec(folding).reshape(-1, 1)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return two_state_noneq_folding_implicit_vec(folding).reshape(-1, 1)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            return two_state_noneq_folding_ode_vec(folding).reshape(-1, 1)
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return three_state_noneq_folding_implicit_vec(folding, synthesis).reshape(-1, 1)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            return three_state_noneq_folding_ode_vec(folding, synthesis).reshape(-1, 1)
        else:
            raise ValueError('model_type does not exist')


class StateProbBound(hk.Module):
    def __init__(self, model_type='tri_state_equilibrium_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, binding, folding, synthesis=None, degradation=None):
        if self.model_type == 'tri_state_equilibrium_explicit':
         return 1/(1+jnp.exp(binding)*(1+jnp.exp(folding)))
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return opt_3st_vec(binding, folding).reshape(-1, 1)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return ss_tri_state_vec(binding, folding).reshape(-1, 1)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return two_state_noneq_binding_implicit_vec(binding, folding, degradation).reshape(-1, 1)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            return two_state_noneq_binding_ode_vec(binding, folding, degradation).reshape(-1, 1)
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return three_state_noneq_binding_implicit_vec(binding, folding, synthesis, degradation).reshape(-1, 1)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            return three_state_noneq_binding_ode_vec(binding, folding, synthesis, degradation).reshape(-1, 1)
        else:
            raise ValueError('model type does not exist')
