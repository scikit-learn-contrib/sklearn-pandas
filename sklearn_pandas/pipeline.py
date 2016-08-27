import six
import collections
from sklearn.pipeline import Pipeline, FeatureUnion, _name_estimators
from sklearn.utils import tosequence

from .utils import PassThroughTransformer, ColumnSelectTransformer


def _call_fit(fit_method, X, y=None, **kwargs):
    """
    helper function, calls the fit or fit_transform method with the correct
    number of parameters

    fit_method: fit or fit_transform method of the transformer
    X: the data to fit
    y: the target vector relative to X, optional
    kwargs: any keyword arguments to the fit method

    return: the result of the fit or fit_transform method

    WARNING: if this function raises a TypeError exception, test the fit
    or fit_transform method passed to it in isolation as _call_fit will not
    distinguish TypeError due to incorrect number of arguments from
    other TypeError
    """
    try:
        return fit_method(X, y, **kwargs)
    except TypeError:
        # fit takes only one argument
        return fit_method(X, **kwargs)


class TransformerPipeline(Pipeline):
    """
    Pipeline that expects all steps to be transformers taking a single X argument,
    an optional y argument,
    and having fit and transform methods.

    Code is copied from sklearn's Pipeline
    """
    def __init__(self, steps):
        names, estimators = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError("Provided step names are not unique: %s" % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        estimator = estimators[-1]

        for e in estimators:
            if (not (hasattr(e, "fit") or hasattr(e, "fit_transform")) or not
                    hasattr(e, "transform")):
                raise TypeError("All steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (e, type(e)))

        if not hasattr(estimator, "fit"):
            raise TypeError("Last step of chain should implement fit "
                            "'%s' (type %s) doesn't)"
                            % (estimator, type(estimator)))

    # needed for compatibility with sklearn<=0.16, that doesn't have
    # this property defined
    @property
    def named_steps(self):
        return dict(self.steps)

    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = _call_fit(transform.fit_transform,
                               Xt, y, **fit_params_steps[name])
            else:
                Xt = _call_fit(transform.fit,
                               Xt, y, **fit_params_steps[name]).transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        _call_fit(self.steps[-1][-1].fit, Xt, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return _call_fit(self.steps[-1][-1].fit_transform,
                             Xt, y, **fit_params)
        else:
            return _call_fit(self.steps[-1][-1].fit,
                             Xt, y, **fit_params).transform(Xt)


def make_transformer_pipeline(feature_selector, transformers, feature_name):
    if not isinstance(transformers, list):
        transformers = [transformers]
    # transform None into PassThroughTransformer
    transformers = [PassThroughTransformer() if t is None else t
                    for t in transformers]
    cst = ColumnSelectTransformer
    selector = [('selector', cst(feature_selector))]
    # pipeline of selector followed by transformer
    pipe = TransformerPipeline(selector + _name_estimators(transformers))
    return (feature_name, pipe)


def make_feature_union(mapping, n_jobs=1):
    """
    Create a FeatureUnion from the specified mapping.

    Creates a FeatureUnion of TransformerPipelines that select the columns
    given in the mapping as first step, then apply the specified transformers
    sequentially.

    :param mapping: a list of tuples where the first is the column name(s) and
        the other is the transormation or list of transformation to apply.
        See ``DataFrameMapper`` for more information.
    :param n_jobs: number of jobs to run in parallel (default 1)
    """
    feature_names = get_feature_names(mapping)
    # repeated feature names are not allowed, since they collide when
    # doing set_params()
    dupe_feat_names = [item for item, count in
                       collections.Counter(feature_names).items() if count > 1]
    if len(dupe_feat_names):
        raise ValueError(
            'Duplicated feature column names found: {}. Please '
            'provide custom feature names to '
            'disambiguate.'.format(dupe_feat_names))

    feature_selectors = [el[0] for el in mapping]
    transformers = [el[1] for el in mapping]
    transformer_pipes = [
        make_transformer_pipeline(feature_selector, transformers, feature_name)
        for feature_selector, transformers, feature_name in
        zip(feature_selectors, transformers, feature_names)]

    if transformer_pipes:  # at least one column to be transformed
        feature_union = FeatureUnion(transformer_pipes, n_jobs=n_jobs)
    else:  # case when no columns were selected, but specifying default
        feature_union = None
    return feature_union


def get_feature_names(mapping):
    """
    Derive feature names from given feature definition mapping.

    By default, it takes the string representation of selected column(s) names,
    but a custom name can be provided as the third argument of the feature
    definition tuple.
    """
    return [feat_def[2] if len(feat_def) == 3 else str(feat_def[0])
            for feat_def in mapping]