#!/usr/bin/env python

from setuptools import setup

import sklearn_pandas

setup(name='sklearn-pandas',
      version=sklearn_pandas.__version__,
      description='Pandas integration with sklearn',
      author='Paul Butler',
      author_email='paulgb@gmail.com',
      url='https://github.com/paulgb/sklearn-pandas',
      packages=['sklearn_pandas'],
      keywords=['scikit', 'sklearn', 'pandas'],
      install_requires=[
          'scikit-learn>=0.14',
          'pandas>=0.11.0',
          'numpy>=1.6.1']
)

