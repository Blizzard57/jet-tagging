rm -r build/ dist/ phymlq.egg-info
python3 setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python3 -m pip install "dist/phymlq-0.1.1-py3-none-any.whl" --force
python3 tests/test.py

python3 -m twine upload --repository testpypi dist/*
