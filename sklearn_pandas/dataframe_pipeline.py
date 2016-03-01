import six
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method

from sklearn_pandas.dataframe_mapper import DataFrameMapper

class DataFramePipeline(Pipeline):
    """Pipeline of transforms with a final estimator supporting DataFrame input.
    
    Sequentially applies a list of transforms and a final estimator, extracting
    'X' and 'y' data for the pipeline via an initial DataFrameMapper. The first
    step of the pipeline must be a DataFrameMapper. Intermediate steps of the
    pipeline must be 'transforms' implementing the 'fit' and 'transform' methods.
    The final estimator only needs to implement fit.
    
    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    """
    
    def __init__(self, steps):
        Pipeline.__init__(self, steps)
        
        if not isinstance(self._dataframe_mapper, DataFrameMapper):
            raise TypeError(
                "First step of a DataFramePipeline must be a DataFrameMapper, "
                "'%s' (type %s) is not." %
                (self._dataframe_mapper, type(self._dataframe_mapper))
            )
            
    @property
    def _dataframe_mapper(self):
        """Return DataFrameMapper at head of pipeline."""
        return self.steps[0][1]
    
    @property
    def _dataframe_mapper_name(self):
        """Return name of DataFrameMapper at head of pipeline."""
        return self.steps[0][0]
    
    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
            
        Xt = self._dataframe_mapper.fit_transform(
            X, y, **fit_params_steps[self._dataframe_mapper_name])
        yt = self._dataframe_mapper.extract_y(X, y)
        
        for name, transform in self.steps[1:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]) \
                              .transform(Xt)
                    
        return Xt, yt, fit_params_steps[self.steps[-1][0]]
    
    def fit(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : (DataFrame)
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : (DataFrame or Series), default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline. Training target are extracted via mapper from 'X' if
            'y' is None.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        self.steps[-1][-1].fit(Xt, yt, **fit_params)
        return self
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator.

        Parameters
        ----------
        X : (DataFrame)
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : (DataFrame or Series), default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline. Training target are extracted via mapper from 'X' if
            'y' is None.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Xt, yt, **fit_params)
        else:
            return self.steps[-1][-1].fit(Xt, yt, **fit_params).transform(Xt)
        
    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : (DataFrame)
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : (DataFrame or Series), default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline. Training target are extracted via mapper from 'X' if
            'y' is None.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)
    
    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score.

        Parameters
        ----------
        X : (DataFrame)
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : (DataFrame or Series), default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline. Training target are extracted via mapper from 'X' if
            'y' is None.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
            
        yt = self._dataframe_mapper.extract_y(X, y)
        return self.steps[-1][-1].score(Xt, yt)

def make_dataframe_pipeline(steps):
    """Construct a DataFramePipeline from the given estimators."""
    return DataFramePipeline(_name_estimators(steps))
