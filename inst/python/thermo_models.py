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
            return self._tri_state_explicit(folding)
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return self._implicit_layers(folding)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return self._ODE_layers(folding)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return self._two_state_non_equilibrium_implicit(folding)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            return self._two_state_non_equilibrium_ODE(folding)
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return self._tri_state_non_equilibrium_implicit(folding, synthesis)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            return self._tri_state_non_equilibrium_ODE(folding, synthesis)
        else:
            raise ValueError('model_type does not exist')

    def _tri_state_explicit(self, folding):
        return 1/(1+jnp.exp(folding))

    def _implicit_layers(self, folding):
        return opt_2st_vec(folding).reshape(-1, 1)

    def _ODE_layers(self, folding):
        return ss_two_state_vec(folding).reshape(-1, 1)

    def _two_state_non_equilibrium_implicit(self, folding):
        return two_state_noneq_folding_implicit_vec(folding).reshape(-1, 1)

    def _two_state_non_equilibrium_ODE(self, folding):
        return two_state_noneq_folding_ode_vec(folding).reshape(-1, 1)

    def _tri_state_non_equilibrium_implicit(self, folding, synthesis):
        return three_state_noneq_folding_implicit_vec(folding, synthesis).reshape(-1, 1)

    def _tri_state_non_equilibrium_ODE(self, folding, synthesis):
        return three_state_noneq_folding_ode_vec(folding, synthesis).reshape(-1, 1)


class StateProbBound(hk.Module):
    def __init__(self, model_type='tri_state_equilibrium_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, binding, folding, synthesis=None, degradation=None):
        if self.model_type == 'tri_state_equilibrium_explicit':
            return self._tri_state_explicit(binding, folding)
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return self._implicit_layers(binding, folding)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return self._ODE_layers(binding, folding)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return self._two_state_non_equilibrium_implicit(binding, folding, degradation)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            return self._two_state_non_equilibrium_ODE(binding, folding, degradation)
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return self._tri_state_non_equilibrium_implicit(binding, folding, synthesis, degradation)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            return self._tri_state_non_equilibrium_ODE(binding, folding, synthesis, degradation)
        else:
            raise ValueError('model type does not exist')

    def _tri_state_explicit(self, binding, folding):
        return 1/(1+jnp.exp(binding)*(1+jnp.exp(folding)))

    def _implicit_layers(self, binding, folding):
        return opt_3st_vec(binding, folding).reshape(-1, 1)

    def _ODE_layers(self, binding, folding):
        return ss_tri_state_vec(binding, folding).reshape(-1, 1)

    def _two_state_non_equilibrium_implicit(self, binding, folding, degradation):
        #change results processing to only keep bound proportion
        return two_state_noneq_binding_implicit_vec(binding, folding, degradation).reshape(-1, 1)

    def _two_state_non_equilibrium_ODE(self, binding, folding, degradation):
        return two_state_noneq_binding_ode_vec(binding, folding, degradation).reshape(-1, 1)

    def _tri_state_non_equilibrium_implicit(self, binding, folding, synthesis, degradation):
        #change results processing to only keep bound proportion
        return three_state_noneq_binding_implicit_vec(binding, folding, synthesis, degradation).reshape(-1, 1)

    def _tri_state_non_equilibrium_ODE(self, binding, folding, synthesis, degradation):
        return three_state_noneq_binding_ode_vec(binding, folding, synthesis, degradation).reshape(-1, 1)
