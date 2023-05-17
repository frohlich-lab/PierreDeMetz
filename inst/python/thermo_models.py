import jax.numpy as jnp
import haiku as hk
from chem_model_3eq import opt_2st_vec, opt_3st_vec, ss_two_state_vec, ss_tri_state_vec
from chem_model_2neq import two_state_noneq_folding_implicit_vec, two_state_noneq_binding_implicit_vec
#from chem_model_3neq import three_state_noneq_binding_implicit_vec, three_state_noneq_folding_implicit_vec

##################### IMPLEMENTATION OF THE THREE MODELS #####################
class StateProbFolded(hk.Module):
    def __init__(self, model_type='tri_state_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, inputs_1, inputs_2):
        if self.model_type == 'tri_state_equilibrium_explicit':
            return self._tri_state_explicit(inputs_2)
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return self._implicit_layers(inputs_2)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return self._ODE_layers(inputs_2)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return self._two_state_non_equilibrium_implicit(inputs_2)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            pass
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return self._tri_state_non_equilibrium_implicit(inputs_2)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            pass
        else:
            raise ValueError('model_type does not exist')

    def _tri_state_explicit(self, inputs_2):
        return 1/(1+jnp.exp(inputs_2))

    def _implicit_layers(self, inputs_2):
        return opt_2st_vec(inputs_2).reshape(-1, 1)

    def _ODE_layers(self, inputs_2):
        return ss_two_state_vec(inputs_2).reshape(-1, 1)

    def _two_state_non_equilibrium_implicit(self, inputs_2):
        #change results processing to only keep folded proportion
        return two_state_noneq_folding_implicit_vec(inputs_2).reshape(-1, 1)

    def _two_state_non_equilibrium_ODE(self, inputs):
        pass

    def _tri_state_non_equilibrium_implicit(self, inputs_1, inputs_2):
        #change results processing to only keep folded proportion
        return three_state_noneq_folding_implicit_vec(inputs_1, inputs_2)[2]

    def _tri_state_non_equilibrium_ODE(self, inputs):
        pass


class StateProbBound(hk.Module):
    def __init__(self, model_type='tri_state_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, inputs_1, inputs_2):
        if self.model_type == 'tri_state_equilibrium_explicit':
            return self._tri_state_explicit(inputs_1, inputs_2)
        elif self.model_type == 'tri_state_equilibrium_implicit':
            return self._implicit_layers(inputs_1, inputs_2)
        elif self.model_type == 'tri_state_equilibrium_ODE':
            return self._ODE_layers(inputs_1, inputs_2)
        elif self.model_type == 'two_state_non_equilibrium_implicit':
            return self._two_state_non_equilibrium_implicit(inputs_1, inputs_2)
        elif self.model_type == 'two_state_non_equilibrium_ODE':
            pass
        elif self.model_type == 'tri_state_non_equilibrium_implicit':
            return self._tri_state_non_equilibrium_implicit(inputs_1, inputs_2)
        elif self.model_type == 'tri_state_non_equilibrium_ODE':
            pass
        else:
            raise ValueError('model type does not exist')

    def _tri_state_explicit(self, inputs_1, inputs_2):
        return 1/(1+jnp.exp(inputs_1)*(1+jnp.exp(inputs_2)))

    def _implicit_layers(self, inputs_1, inputs_2):
        return opt_3st_vec(inputs_1, inputs_2).reshape(-1, 1)

    def _ODE_layers(self, inputs_1, inputs_2):
        return ss_tri_state_vec(inputs_1, inputs_2).reshape(-1, 1)

    def _two_state_non_equilibrium_implicit(self, inputs_1, inputs_2):
        #change results processing to only keep bound proportion
        return two_state_noneq_binding_implicit_vec(inputs_1, inputs_2).reshape(-1, 1)

    def _two_state_non_equilibrium_ODE(self, inputs_1, inputs_2):
        pass

    def _tri_state_non_equilibrium_implicit(self, inputs_1, inputs_2):
        #change results processing to only keep bound proportion
        return three_state_noneq_binding_implicit_vec(inputs_1, inputs_2)[1]

    def _tri_state_non_equilibrium_ODE(self, inputs_1, inputs_2):
        pass
