
Sklearn-pandas
==============

This module provides a bridge between `Scikit-Learn <http://scikit-learn.org/stable/>`__'s machine learning methods and `pandas <http://pandas.pydata.org/>`__-style Data Frames.

In particular, it provides:

1. A way to map ``DataFrame`` columns to transformations, which are later recombined into features.
2. A compatibility shim for old ``scikit-learn`` versions to cross-validate a pipeline that takes a pandas ``DataFrame`` as input. This is only needed for ``scikit-learn<0.16.0`` (see `#11 <https://github.com/paulgb/sklearn-pandas/issues/11>`__ for details). It is deprecated and will likely be dropped in ``skearn-pandas==2.0``.

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
* ``cross_val_score``, similar to ``sklearn.cross_validation.cross_val_score`` but working on pandas DataFrames

For this demonstration, we will import both::

    >>> from sklearn_pandas import DataFrameMapper, cross_val_score

For these examples, we'll also use pandas, numpy, and sklearn::

    >>> import pandas as pd
    >>> import numpy as np
    >>> import sklearn.preprocessing, sklearn.decomposition, \
    ...     sklearn.linear_model, sklearn.pipeline, sklearn.metrics
    >>> from sklearn.feature_extraction.text import CountVectorizer

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

The mapper takes a list of pairs. The first is a column name from the pandas DataFrame, or a list containing one or multiple columns (we will see an example with multiple columns later). The second is an object which will perform the transformation which will be applied to that column::

    >>> mapper = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     (['children'], sklearn.preprocessing.StandardScaler())
    ... ])

The difference between specifying the column selector as ``'column'`` (as a simple string) and ``['column']`` (as a list with one element) is the shape of the array that is passed to the transformer. In the first case, a one dimensional array with be passed, while in the second case it will be a 2-dimensional array with one column, i.e. a column vector.

This behaviour mimics the same pattern as pandas' dataframes ``__getitem__``  indexing:

    >>> data['children'].shape
    (8,)
    >>> data[['children']].shape
    (8, 1)

Be aware that some transformers expect a 1-dimensional input (the label-oriented ones) while some others, like ``OneHotEncoder`` or ``Imputer``, expect 2-dimensional input, with the shape ``[n_samples, n_features]``.

Test the Transformation
***********************

We can use the ``fit_transform`` shortcut to both fit the model and see what transformed data looks like. In this and the other examples, output is rounded to two digits with ``np.round`` to account for rounding errors on different hardware::

    >>> np.round(mapper.fit_transform(data.copy()), 2)
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

After transformation, the ``feature_indices_` attribute of the mapper
indicates which columns of the resulting output array correspond to which
input features. Input feature ``i`` is mapped to features from
``feature_indices_[i]`` to ``feature_indices_[i+1]`` in transformed output.
For example:

    >>> mapper.feature_indices_[0], mapper.feature_indices_[1] # pet
    (0, 3)
    >>> mapper.feature_indices_[1], mapper.feature_indices_[2]  # children
    (3, 4)

Transform Multiple Columns
**************************

Transformations may require multiple input columns. In these cases, the column names can be specified in a list::

    >>> mapper2 = DataFrameMapper([
    ...     (['children', 'salary'], sklearn.decomposition.PCA(1))
    ... ])

Now running ``fit_transform`` will run PCA on the ``children`` and ``salary`` columns and return the first principal component::

    >>> np.round(mapper2.fit_transform(data.copy()), 1)
    array([[ 47.6],
           [-18.4],
           [  1.6],
           [-15.4],
           [-10.4],
           [ 16.6],
           [ -6.4],
           [-15.4]])

Multiple transformers for the same column
*****************************************

Multiple transformers can be applied to the same column specifying them
in a list::

    >>> mapper3 = DataFrameMapper([
    ...     (['age'], [sklearn.preprocessing.Imputer(),
    ...                sklearn.preprocessing.StandardScaler()])])
    >>> data_3 = pd.DataFrame({'age': [1, np.nan, 3]})
    >>> mapper3.fit_transform(data_3)
    array([[-1.22474487],
           [ 0.        ],
           [ 1.22474487]])

Columns that don't need any transformation
******************************************

Only columns that are listed in the DataFrameMapper are kept. To keep a column but don't apply any transformation to it, use `None` as transformer::

    >>> mapper3 = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     ('children', None)
    ... ])
    >>> np.round(mapper3.fit_transform(data.copy()))
    array([[ 1.,  0.,  0.,  4.],
           [ 0.,  1.,  0.,  6.],
           [ 0.,  1.,  0.,  3.],
           [ 0.,  0.,  1.,  3.],
           [ 1.,  0.,  0.,  2.],
           [ 0.,  1.,  0.,  3.],
           [ 1.,  0.,  0.,  5.],
           [ 0.,  0.,  1.,  4.]])


Working with sparse features
****************************

``DataFrameMapper``s will return a dense feature array by default. Setting ``sparse=True`` in the mapper will return a sparse array whenever any of the extracted features is sparse. Example:

    >>> mapper4 = DataFrameMapper([
    ...     ('pet', CountVectorizer()),
    ... ], sparse=True)
    >>> type(mapper4.fit_transform(data))
    <class 'scipy.sparse.csr.csr_matrix'>

The stacking of the sparse features is done without ever densifying them.

Cross-Validation
----------------

Now that we can combine features from pandas DataFrames, we may want to use cross-validation to see whether our model works. ``scikit-learn<0.16.0`` provided features for cross-validation, but they expect numpy data structures and won't work with ``DataFrameMapper``.

To get around this, sklearn-pandas provides a wrapper on sklearn's ``cross_val_score`` function which passes a pandas DataFrame to the estimator rather than a numpy array::

    >>> pipe = sklearn.pipeline.Pipeline([
    ...     ('featurize', mapper),
    ...     ('lm', sklearn.linear_model.LinearRegression())])
    >>> np.round(cross_val_score(pipe, data.copy(), data.salary, 'r2'), 2)
    array([ -1.09,  -5.3 , -15.38])

Sklearn-pandas' ``cross_val_score`` function provides exactly the same interface as sklearn's function of the same name.


Changelog
---------

Development
***********

* Deprecate custom cross-validation shim classes.
* Require ``scikit-learn>=0.15.0``. Resolves #49.
* Add ``feature_indices_`` attribute indicating the mapping between input and
  ouptut variables.


1.1.0 (2015-12-06)
*******************

* Delete obsolete ``PassThroughTransformer``. If no transformation is desired for a given column, use ``None`` as transformer.
* Factor out code in several modules, to avoid having everything in ``__init__.py``.
* Use custom ``TransformerPipeline`` class to allow transformation steps accepting only a X argument. Fixes #46.
* Add compatibility shim for unpickling mappers with list of transformers created before 1.0.0. Fixes #45.


1.0.0 (2015-11-28)
*******************

* Change version numbering scheme to SemVer.
* Use ``sklearn.pipeline.Pipeline`` instead of copying its code. Resolves #43.
* Raise ``KeyError`` when selecting unexistent columns in the dataframe. Fixes #30.
* Return sparse feature array if any of the features is sparse and ``sparse`` argument is ``True``. Defaults to ``False`` to avoid potential breaking of existing code. Resolves #34.
* Return model and prediction in custom CV classes. Fixes #27.


0.0.12 (2015-11-07)
********************

* Allow specifying a list of transformers to use sequentially on the same column.


Credits
-------

The code for ``DataFrameMapper`` is based on code originally written by `Ben Hamner <https://github.com/benhamner>`__.

Other contributors:

* Paul Butler
* Cal Paterson
* Israel Saeta Pérez
* Zac Stewart
* Olivier Grisel
