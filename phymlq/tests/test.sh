rm -r build/ dist/ phymlq.egg-info
python setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python -m pip install "dist/phymlq-0.0.1a0-py3-none-any.whl" --force
python tests/test.py

python -m twine upload --repository testpypi dist/*