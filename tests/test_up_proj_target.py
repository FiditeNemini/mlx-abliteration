import numpy as np

from tests.helpers.shims import install_shims

# install shims for this test
install_shims()

from core.abliteration import get_ablated_parameters


class DummyModelWithUpProj:
    def __init__(self):
        self.parameters_dict = {
            'model.layers.0.mlp.up_proj.weight': np.array([[0.2, 0.1], [0.0, 0.3]]),
            'model.layers.0.mlp.c_proj.weight': np.array([[1.0, 0.0], [0.0, 1.0]]),
        }

    def parameters(self):
        return list(self.parameters_dict.items())

    def update(self, params):
        for k, v in params.items():
            self.parameters_dict[k] = v


def test_up_proj_is_targeted():
    model = DummyModelWithUpProj()
    rv = np.array([1.0, 0.0])
    # Pass explicit target_modules including mlp.up_proj
    new_params = get_ablated_parameters(model, rv, target_modules=["mlp.up_proj"], ablation_strength=1.0)
    assert 'model.layers.0.mlp.up_proj.weight' in new_params
