rm -r build/ dist/ phymlq.egg-info
python setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python -m pip install "dist/phymlq-0.0.0a0-py3-none-any.whl" --force
python -c "import phymlq; phymlq.ml.particle_net.weights.say_hello();"
python tests/test.py

python -m twine upload --repository testpypi dist/*