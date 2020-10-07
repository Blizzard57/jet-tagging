import phymlq
import numpy as np
import tensorflow as tf

def test_particlenet_model(self):
    dataset = {
        'points': np.random.random((25, 100, 2)),
        'features': np.random.random((25, 100, 4)),
        'mask': np.random.random((25, 100, 1))
    }
    input_shapes = {k : dataset[k].shape[1:] for k in dataset}

    model = phymlq.hep.particle_net.models.ParticleNet(input_shapes, 2)
    assert hasattr(model, 'summary')
