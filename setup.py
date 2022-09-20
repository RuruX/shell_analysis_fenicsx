from distutils.core import setup
import setuptools
setup(
    name='shell_analysis_fenicsX',
    version='0.1',
    packages=[
        'shell_analysis_fenicsX'
    ],
    url='https://github.com/RuruX/shell-analysis-fenicsx',
    author='Ru Xiang',
    author_email='rxiang@ucsd.edu',
    description="Shell model analysis with FEniCSx",
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
