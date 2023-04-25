import jax.numpy as jnp
import haiku as hk

##################### IMPLEMENTATION OF THE THREE MODELS #####################
class StateProbFolded(hk.Module):
    def __init__(self, model_type='tri_state_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, inputs):
        if self.model_type == 'tri_state_explicit':
            return self._tri_state_explicit(inputs)
        elif self.model_type == 'implicit':
            return self._implicit_layers(inputs)
        elif self.model_type == 'ODE':
            return self._ODE_layers(inputs)
        else:
            raise ValueError('model_type must be one of tri_state_explicit, implicit, ODE')

    def _tri_state_explicit(self, inputs):
        return 1/(1+jnp.exp(inputs))

    def _implicit_layers(self, inputs):
        pass

    def _ODE_layers(self, inputs):
        pass

class StateProbFBound(hk.Module):
    def __init__(self, model_type='tri_state_explicit'):
        super().__init__()
        self.model_type = model_type

    def __call__(self, inputs_1, inputs_2):
        if self.model_type == 'tri_state_explicit':
            return self._tri_state_explicit(inputs_1, inputs_2)
        elif self.model_type == 'implicit':
            return self._implicit_layers(inputs_1, inputs_2)
        elif self.model_type == 'ODE':
            return self._ODE_layers(inputs_1, inputs_2)
        else:
            raise ValueError('model_type must be one of tri_state_explicit, implicit, ODE')

    def _tri_state_explicit(self, inputs_1, inputs_2):
        return 1/(1+jnp.exp(inputs_1)*(1+jnp.exp(inputs_2)))

    def _implicit_layers(self, inputs_1, inputs_2):
        pass

    def _ODE_layers(self, inputs_1, inputs_2):
        pass
