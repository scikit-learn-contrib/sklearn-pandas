import six
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.utils import tosequence


class TransformerPipeline(Pipeline):
    """
    Pipeline that expects all steps to be transformers taking a single argument
    and having fit and transform methods.

    Code is copied from sklearn's Pipeline, leaving out the `y=None` argument.
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

    def _pre_transform(self, X, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, **fit_params_steps[name]) \
                              .transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, **fit_params):
        Xt, fit_params = self._pre_transform(X, **fit_params)
        self.steps[-1][-1].fit(Xt, **fit_params)
        return self

    def fit_transform(self, X, **fit_params):
        Xt, fit_params = self._pre_transform(X, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Xt, **fit_params)
        else:
            return self.steps[-1][-1].fit(Xt, **fit_params).transform(Xt)


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return TransformerPipeline(_name_estimators(steps))
