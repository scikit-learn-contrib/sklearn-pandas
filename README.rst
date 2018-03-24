
Sklearn-pandas
==============

.. image:: https://circleci.com/gh/pandas-dev/sklearn-pandas.svg?style=svg
    :target: https://circleci.com/gh/pandas-dev/sklearn-pandas

This module provides a bridge between `Scikit-Learn <http://scikit-learn.org/stable>`__'s machine learning methods and `pandas <https://pandas.pydata.org>`__-style Data Frames.

In particular, it provides:

1. A way to map ``DataFrame`` columns to transformations, which are later recombined into features.
2. A compatibility shim for old ``scikit-learn`` versions to cross-validate a pipeline that takes a pandas ``DataFrame`` as input. This is only needed for ``scikit-learn<0.16.0`` (see `#11 <https://github.com/paulgb/sklearn-pandas/issues/11>`__ for details). It is deprecated and will likely be dropped in ``skearn-pandas==2.0``.
3. A ``CategoricalImputer`` that replaces null-like values with the mode and works with string columns.

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

Normally you'll read the data from a file, but for demonstration purposes we'll create a data frame from a Python dict::

    >>> data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
    ...                      'children': [4., 6, 3, 3, 2, 3, 5, 4],
    ...                      'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})

Transformation Mapping
----------------------

Map the Columns to Transformations
**********************************

The mapper takes a list of tuples. The first is a column name from the pandas DataFrame, or a list containing one or multiple columns (we will see an example with multiple columns later). The second is an object which will perform the transformation which will be applied to that column. The third is optional and is a dictionary containing the transformation options, if applicable (see "custom column names for transformed features" below).

Let's see an example::

    >>> mapper = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     (['children'], sklearn.preprocessing.StandardScaler())
    ... ])

The difference between specifying the column selector as ``'column'`` (as a simple string) and ``['column']`` (as a list with one element) is the shape of the array that is passed to the transformer. In the first case, a one dimensional array will be passed, while in the second case it will be a 2-dimensional array with one column, i.e. a column vector.

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

Note that the first three columns are the output of the ``LabelBinarizer`` (corresponding to ``cat``, ``dog``, and ``fish`` respectively) and the fourth column is the standardized value for the number of children. In general, the columns are ordered according to the order given when the ``DataFrameMapper`` is constructed.

Now that the transformation is trained, we confirm that it works on new data::

    >>> sample = pd.DataFrame({'pet': ['cat'], 'children': [5.]})
    >>> np.round(mapper.transform(sample), 2)
    array([[ 1.  ,  0.  ,  0.  ,  1.04]])


Output features names
*********************

In certain cases, like when studying the feature importances for some model,
we want to be able to associate the original features to the ones generated by
the dataframe mapper. We can do so by inspecting the automatically generated ``transformed_names_`` attribute of the mapper after transformation::

    >>> mapper.transformed_names_
    ['pet_cat', 'pet_dog', 'pet_fish', 'children']


Custom column names for transformed features
********************************************

We can provide a custom name for the transformed features, to be used instead
of the automatically generated one, by specifying it as the third argument
of the feature definition::


  >>> mapper_alias = DataFrameMapper([
  ...     (['children'], sklearn.preprocessing.StandardScaler(),
  ...      {'alias': 'children_scaled'})
  ... ])
  >>> _ = mapper_alias.fit_transform(data.copy())
  >>> mapper_alias.transformed_names_
  ['children_scaled']


Passing Series/DataFrames to the transformers
*********************************************

By default the transformers are passed a numpy array of the selected columns
as input. This is because ``sklearn`` transformers are historically designed to
work with numpy arrays, not with pandas dataframes, even though their basic
indexing interfaces are similar.

However we can pass a dataframe/series to the transformers to handle custom
cases initializing the dataframe mapper with ``input_df=True``::

    >>> from sklearn.base import TransformerMixin
    >>> class DateEncoder(TransformerMixin):
    ...    def fit(self, X, y=None):
    ...        return self
    ...
    ...    def transform(self, X):
    ...        dt = X.dt
    ...        return pd.concat([dt.year, dt.month, dt.day], axis=1)
    >>> dates_df = pd.DataFrame(
    ...     {'dates': pd.date_range('2015-10-30', '2015-11-02')})
    >>> mapper_dates = DataFrameMapper([
    ...     ('dates', DateEncoder())
    ... ], input_df=True)
    >>> mapper_dates.fit_transform(dates_df)
    array([[2015,   10,   30],
           [2015,   10,   31],
           [2015,   11,    1],
           [2015,   11,    2]])

We can also specify this option per group of columns instead of for the
whole mapper::

  >>> mapper_dates = DataFrameMapper([
  ...     ('dates', DateEncoder(), {'input_df': True})
  ... ])
  >>> mapper_dates.fit_transform(dates_df)
  array([[2015,   10,   30],
         [2015,   10,   31],
         [2015,   11,    1],
         [2015,   11,    2]])

Outputting a dataframe
**********************

By default the output of the dataframe mapper is a numpy array. This is so because most sklearn estimators expect a numpy array as input. If however we want the output of the mapper to be a dataframe, we can do so using the parameter ``df_out`` when creating the mapper::

    >>> mapper_df = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     (['children'], sklearn.preprocessing.StandardScaler())
    ... ], df_out=True)
    >>> np.round(mapper_df.fit_transform(data.copy()), 2)
       pet_cat  pet_dog  pet_fish  children
    0      1.0      0.0       0.0      0.21
    1      0.0      1.0       0.0      1.88
    2      0.0      1.0       0.0     -0.63
    3      0.0      0.0       1.0     -0.63
    4      1.0      0.0       0.0     -1.46
    5      0.0      1.0       0.0     -0.63
    6      1.0      0.0       0.0      1.04
    7      0.0      0.0       1.0      0.21

The names for the columns are the same ones present in the ``transformed_names_``
attribute.

Note this does not work together with the ``default=True`` or ``sparse=True`` arguments to the mapper.

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

Applying a default transformer
******************************

A default transformer can be applied to columns not explicitly selected
passing it as the ``default`` argument to the mapper:

    >>> mapper4 = DataFrameMapper([
    ...     ('pet', sklearn.preprocessing.LabelBinarizer()),
    ...     ('children', None)
    ... ], default=sklearn.preprocessing.StandardScaler())
    >>> np.round(mapper4.fit_transform(data.copy()), 1)
    array([[ 1. ,  0. ,  0. ,  4. ,  2.3],
           [ 0. ,  1. ,  0. ,  6. , -0.9],
           [ 0. ,  1. ,  0. ,  3. ,  0.1],
           [ 0. ,  0. ,  1. ,  3. , -0.7],
           [ 1. ,  0. ,  0. ,  2. , -0.5],
           [ 0. ,  1. ,  0. ,  3. ,  0.8],
           [ 1. ,  0. ,  0. ,  5. , -0.3],
           [ 0. ,  0. ,  1. ,  4. , -0.7]])

Using ``default=False`` (the default) drops unselected columns. Using
``default=None`` pass the unselected columns unchanged.


Same transformer for the multiple columns
*****************************************

Sometimes it is required to apply the same transformation to several dataframe columns.
To simplify this process, the package provides ``gen_features`` function which accepts a list
of columns and feature transformer class (or list of classes), and generates a feature definition,
acceptable by ``DataFrameMapper``.

For example, consider a dataset with three categorical columns, 'col1', 'col2', and 'col3',
To binarize each of them, one could pass column names and ``LabelBinarizer`` transformer class
into generator, and then use returned definition as ``features`` argument for ``DataFrameMapper``:

    >>> from sklearn_pandas import gen_features
    >>> feature_def = gen_features(
    ...     columns=['col1', 'col2', 'col3'],
    ...     classes=[sklearn.preprocessing.LabelEncoder]
    ... )
    >>> feature_def
    [('col1', [LabelEncoder()]), ('col2', [LabelEncoder()]), ('col3', [LabelEncoder()])]
    >>> mapper5 = DataFrameMapper(feature_def)
    >>> data5 = pd.DataFrame({
    ...     'col1': ['yes', 'no', 'yes'],
    ...     'col2': [True, False, False],
    ...     'col3': ['one', 'two', 'three']
    ... })
    >>> mapper5.fit_transform(data5)
    array([[1, 1, 0],
           [0, 0, 2],
           [1, 0, 1]])

If it is required to override some of transformer parameters, then a dict with 'class' key and
transformer parameters should be provided. For example, consider a dataset with missing values.
Then the following code could be used to override default imputing strategy:

    >>> feature_def = gen_features(
    ...     columns=[['col1'], ['col2'], ['col3']],
    ...     classes=[{'class': sklearn.preprocessing.Imputer, 'strategy': 'most_frequent'}]
    ... )
    >>> mapper6 = DataFrameMapper(feature_def)
    >>> data6 = pd.DataFrame({
    ...     'col1': [None, 1, 1, 2, 3],
    ...     'col2': [True, False, None, None, True],
    ...     'col3': [0, 0, 0, None, None]
    ... })
    >>> mapper6.fit_transform(data6)
    array([[ 1.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.],
           [ 2.,  1.,  0.],
           [ 3.,  1.,  0.]])


Feature selection and other supervised transformations
******************************************************

``DataFrameMapper`` supports transformers that require both X and y arguments. An example of this is feature selection. Treating the 'pet' column as the target, we will select the column that best predicts it.

    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> mapper_fs = DataFrameMapper([(['children','salary'], SelectKBest(chi2, k=1))])
    >>> mapper_fs.fit_transform(data[['children','salary']], data['pet'])
    array([[ 90.],
           [ 24.],
           [ 44.],
           [ 27.],
           [ 32.],
           [ 59.],
           [ 36.],
           [ 27.]])

Working with sparse features
****************************

A ``DataFrameMapper`` will return a dense feature array by default. Setting ``sparse=True`` in the mapper will return a sparse array whenever any of the extracted features is sparse. Example:

    >>> mapper5 = DataFrameMapper([
    ...     ('pet', CountVectorizer()),
    ... ], sparse=True)
    >>> type(mapper5.fit_transform(data))
    <class 'scipy.sparse.csr.csr_matrix'>

The stacking of the sparse features is done without ever densifying them.

Cross-Validation
****************

Now that we can combine features from pandas DataFrames, we may want to use cross-validation to see whether our model works. ``scikit-learn<0.16.0`` provided features for cross-validation, but they expect numpy data structures and won't work with ``DataFrameMapper``.

To get around this, sklearn-pandas provides a wrapper on sklearn's ``cross_val_score`` function which passes a pandas DataFrame to the estimator rather than a numpy array::

    >>> pipe = sklearn.pipeline.Pipeline([
    ...     ('featurize', mapper),
    ...     ('lm', sklearn.linear_model.LinearRegression())])
    >>> np.round(cross_val_score(pipe, X=data.copy(), y=data.salary, scoring='r2'), 2)
    array([ -1.09,  -5.3 , -15.38])

Sklearn-pandas' ``cross_val_score`` function provides exactly the same interface as sklearn's function of the same name.

``CategoricalImputer``
**********************

Since the ``scikit-learn``  ``Imputer`` transformer currently only works with
numbers, ``sklearn-pandas`` provides an equivalent helper transformer that
works with strings, substituting null values with the most frequent value in
that column, or a user-specified value.

Example:

    >>> from sklearn_pandas import CategoricalImputer
    >>> data = np.array(['a', 'b', 'b', np.nan], dtype=object)
    >>> imputer = CategoricalImputer()
    >>> imputer.fit_transform(data)
    array(['a', 'b', 'b', 'b'], dtype=object)


Changelog
---------

1.7.0 (2018-03-24)
******************
* Add ``replacement`` parameter to ``CategoricalImputer`` to allow imputing
  with values other than the mode. (#144)


1.6.0 (2017-10-28)
******************
* Add column name to exception during fit/transform (#110).
* Add ``gen_feature`` helper function to help generating the same transformation for multiple columns (#126).


1.5.0 (2017-06-24)
******************
* Allow inputting a dataframe/series per group of columns.
* Get feature names also from ``estimator.get_feature_names()`` if present.
* Attempt to derive feature names from individual transformers when applying a
  list of transformers.
* Do not mutate features in ``__init__`` to be compatible with
  ``sklearn>=0.20`` (#76).


1.4.0 (2017-05-13)
******************
* Allow specifying a custom name (alias) for transformed columns (#83).
* Capture output columns generated names in ``transformed_names_`` attribute (#78).
* Add ``CategoricalImputer`` that replaces null-like values with the mode
  for string-like columns.
* Add ``input_df`` init argument to allow inputting a dataframe/series to the
  transformers instead of a numpy array (#60).


1.3.0 (2017-01-21)
******************

* Make the mapper return dataframes when ``df_out=True`` (#70, #74).
* Update imports to avoid deprecation warnings in sklearn 0.18 (#68).


1.2.0 (2016-10-02)
******************

* Deprecate custom cross-validation shim classes.
* Require ``scikit-learn>=0.15.0``. Resolves #49.
* Allow applying a default transformer to columns not selected explicitly in
  the mapper. Resolves #55.
* Allow specifying an optional ``y`` argument during transform for
  supervised transformations. Resolves #58.


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

* Arnau Gil Amat (@arnau126)
* Cal Paterson (@calpaterson)
* @defvorfu
* Gustavo Sena Mafra (@gsmafra)
* Israel Saeta Pérez (@dukebody)
* Jeremy Howard (@jph00)
* Jimmy Wan (@jimmywan)
* Olivier Grisel (@ogrisel)
* Paul Butler (@paulgb)
* Ritesh Agrawal (@ragrawal)
* Vitaley Zaretskey (@vzaretsk)
* Zac Stewart (@zacstewart)
* Richard Miller (@rwjmiller)
