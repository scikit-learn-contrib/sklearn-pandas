
Sklearn-pandas
==============

This module provides a bridge between `Scikit-Learn <http://scikit-learn.org/stable/>`__'s machine learning methods and `pandas <http://pandas.pydata.org/>`__-style Data Frames.

In particular, it provides:

1. a way to map DataFrame columns to transformations, which are later recombined into features
2. a way to cross-validate a pipeline that takes a pandas DataFrame as input.

Installation
------------

You can install ``sklearn-pandas`` with ``pip``::

    # pip install sklearn-pandas

Tests
-----

The examples in this file double as basic sanity tests. To run them, use ``doctest``, which is included with python::

    # python -m doctest README.rst

Usage
-----

Import
******

Import what you need from the ``sklearn_pandas`` package. The choices are:

* ``DataFrameMapper``, a class for mapping pandas data frame columns to different sklearn transformations
* ``cross_val_score``, similar to `sklearn.cross_validation.cross_val_score` but working on pandas DataFrames

For this demonstration, we will import both::

    >>> from sklearn_pandas import DataFrameMapper, cross_val_score

For these examples, we'll also use pandas, numpy, and sklearn::

    >>> import pandas as pd
    >>> import numpy as np
    >>> import sklearn.preprocessing, sklearn.decomposition, \
    ...     sklearn.linear_model, sklearn.pipeline, sklearn.metrics

Load some Data
**************

Normally you'll read the data from a file, but for demonstration purposes I'll create a data frame from a Python dict::

    >>> data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
    ...                      'children': [4., 6, 3, 3, 2, 3, 5, 4],
    ...                      'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})

Transformation Mapping
----------------------

Map the Columns to Transformations
**********************************

The mapper takes a list of pairs. The first is a column name from the pandas DataFrame (or a list of multiple columns, as we will see later). The second is an object which will perform the transformation which will be applied to that column::

    >>> mapper = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     ('children', sklearn.preprocessing.StandardScaler())
    ... ])


Test the Transformation
***********************

We can use the ``fit_transform`` shortcut to both fit the model and see what transformed data looks like. In this and the other examples, output is rounded to two digits with ``np.round`` to account for rounding errors on different hardware::

    >>> np.round(mapper.fit_transform(data), 2)
    array([[ 1.  ,  0.  ,  0.  ,  0.21],
           [ 0.  ,  1.  ,  0.  ,  1.88],
           [ 0.  ,  1.  ,  0.  , -0.63],
           [ 0.  ,  0.  ,  1.  , -0.63],
           [ 1.  ,  0.  ,  0.  , -1.46],
           [ 0.  ,  1.  ,  0.  , -0.63],
           [ 1.  ,  0.  ,  0.  ,  1.04],
           [ 0.  ,  0.  ,  1.  ,  0.21]])

Note that the first three columns are the output of the ``LabelBinarizer`` (corresponding to _cat_, _dog_, and _fish_ respectively) and the fourth column is the standardized value for the number of children. In general, the columns are ordered according to the order given when the ``DataFrameMapper`` is constructed.

Now that the transformation is trained, we confirm that it works on new data::

    >>> sample = pd.DataFrame({'pet': ['cat'], 'children': [5.]})
    >>> np.round(mapper.transform(sample), 2)
    array([[ 1.  ,  0.  ,  0.  ,  1.04]])

Transform Multiple Columns
**************************

Transformations may require multiple input columns. In these cases, the column names can be specified in a list::

    >>> mapper2 = DataFrameMapper([
    ...     (['children', 'salary'], sklearn.decomposition.PCA(1))
    ... ])
    
Now running ``fit_transform`` will run PCA on the ``children`` and ``salary`` columns and return the first principal component::

    >>> np.round(mapper2.fit_transform(data), 1)
    array([[ 47.6],
           [-18.4],
           [  1.6],
           [-15.4],
           [-10.4],
           [ 16.6],
           [ -6.4],
           [-15.4]])

Cross-Validation
----------------

Now that we can combine features from pandas DataFrames, we may want to use cross-validation to see whether our model works. Scikit-learn provides features for cross-validation, but they expect numpy data structures and won't work with ``DataFrameMapper``.

To get around this, sklearn-pandas provides a wrapper on sklearn's ``cross_val_score`` function which passes a pandas DataFrame to the estimator rather than a numpy array::

    >>> pipe = sklearn.pipeline.Pipeline([
    ...     ('featurize', mapper),
    ...     ('lm', sklearn.linear_model.LinearRegression())])
    >>> np.round(cross_val_score(pipe, data, data.salary, 'r2'), 2)
    array([ -1.09,  -5.3 , -15.38])

Sklearn-pandas' ``cross_val_score`` function provides exactly the same interface as sklearn's function of the same name.

Credit
------

The code for ``DataFrameMapper`` is based on code originally written by `Ben Hamner <https://github.com/benhamner>`__.

