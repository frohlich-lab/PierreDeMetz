import haiku as hk
from chemical_models import ThreeStateModel

class StateProbFolded(hk.Module):
    def __init__(self, model_type='tri_state_equilibrium_explicit'):
        super().__init__()
        self.model_type = model_type

        # Create ThreeStateModel instance
        if 'ODE' in model_type:
            self.three_state_model = ThreeStateModel(equilibrium='equilibrium' in model_type)

    # Rest of the methods...

    def _ODE_layers(self, folding):
        # Use the ThreeStateModel instance to compute the results
        return self.three_state_model.solve(folding).reshape(-1, 1)

    def _tri_state_non_equilibrium_ODE(self, folding, synthesis):
        # Use the ThreeStateModel instance to compute the results
        return self.three_state_model.solve(folding, synthesis).reshape(-1, 1)


class StateProbBound(hk.Module):
    def __init__(self, model_type='tri_state_equilibrium_explicit'):
        super().__init__()
        self.model_type = model_type

        # Create ThreeStateModel instance
        if 'ODE' in model_type:
            self.three_state_model = ThreeStateModel(equilibrium='equilibrium' in model_type)

    # Rest of the methods...

    def _ODE_layers(self, binding, folding):
        # Use the ThreeStateModel instance to compute the results
        return self.three_state_model.solve(binding, folding).reshape(-1, 1)

    def _tri_state_non_equilibrium_ODE(self, binding, folding, synthesis, degradation):
        # Use the ThreeStateModel instance to compute the results
        return self.three_state_model.solve(binding, folding, synthesis, degradation).reshape(-1, 1)
