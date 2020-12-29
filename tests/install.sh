rm -r build/ dist/ random.egg-info
python3 setup.py bdist_wheel --bdist-dir /tmp/bdistwheel
python3 -m pip install "dist/phymlq-0.2.0-py3-none-any.whl" --force
python3 -m twine upload --repository pypi dist/*
