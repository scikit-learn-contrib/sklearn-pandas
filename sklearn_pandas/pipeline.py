import six
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.utils import tosequence


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


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return TransformerPipeline(_name_estimators(steps))
