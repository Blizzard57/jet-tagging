python setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python -m pip install "dist/mlphile-0.0.1-py3-none-any.whl" --force
python -c "import mlphile as ml; ml.models.particle_net.weights.say_hello();"
python tests/test.py