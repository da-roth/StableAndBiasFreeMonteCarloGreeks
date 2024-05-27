from setuptools import setup, find_packages

setup(
    name='plotsurfacetool',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    author='Daniel Roth',
    author_email='danielroth@posteo.eu',
    description='A package for evaluating and plotting functions.',
    long_description_content_type='text/markdown',
    url='https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/src/PlotSurfaceTool',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
