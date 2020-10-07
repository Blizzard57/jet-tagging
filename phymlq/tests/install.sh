rm -r build/ dist/ phymlq.egg-info
python3 setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python3 -m pip install "dist/phymlq-0.1.1-py3-none-any.whl" --force
python3 -c "import phymlq
import numpy as np
import tensorflow as tf

dataset = {
    'points': np.random.random((25, 100, 2)),
    'features': np.random.random((25, 100, 4)),
    'mask': np.random.random((25, 100, 1))
}
input_shapes = {k : dataset[k].shape[1:] for k in dataset}

model = phymlq.ml.particle_net.models.ParticleNet(input_shapes, 2)
tf.keras.utils.plot_model(model, 'model.png')
"

python3 -m twine upload --repository testpypi dist/*
