#!/usr/bin/env python

from setuptools import setup
import re

for line in open('sklearn_pandas/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()


setup(name='sklearn-pandas',
      version=__version__,
      description='Pandas integration with sklearn',
      maintainer='Cal Paterson',
      maintainer_email='cal@calpaterson.com',
      url='https://github.com/paulgb/sklearn-pandas',
      packages=['sklearn_pandas'],
      keywords=['scikit', 'sklearn', 'pandas'],
      install_requires=[
          'scikit-learn>=0.13',
          'pandas>=0.11.0',
          'numpy>=1.6.1']
)

