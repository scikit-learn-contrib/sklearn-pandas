#!/usr/bin/env python

from setuptools import setup

setup(name='sklearn-pandas',
      version='0.0.3',
      description='Pandas integration with sklearn',
      author='Paul Butler',
      author_email='paulgb@gmail.com',
      url='https://github.com/paulgb/sklearn-pandas',
      packages=['sklearn_pandas'],
      keywords=['scikit', 'sklearn', 'pandas'],
      install_requires=[
          'scikit-learn>=0.13.1',
          'pandas>=0.11.0',
          'numpy>=1.6.1']
)

