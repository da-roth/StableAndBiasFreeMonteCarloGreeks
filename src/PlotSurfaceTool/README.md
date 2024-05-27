# Installation instruction:

# Package local setup
pip install -e . 

# Install twine
pip install twine

# Create wheel from package
python setup.py sdist bdist_wheel

# Go into dist folder
cd dist

# Upload to PyPi
python -m twine upload *