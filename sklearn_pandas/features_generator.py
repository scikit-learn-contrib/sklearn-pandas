def gen_features(columns, classes=None):
    """Generates a feature definition list which can be passed
    into DataFrameMapper

    Params:

    columns     a list of column names to generate features for.

    classes     a list of classes for each feature, a list dictionaries with
                transformer class and init parameters, or None.

                If list of classes is provided, then each of them is
                instantiated with default arguments:

                    classes = [StandardScaler, LabelBinarizer]

                If list of dictionaries is provided, then each of them should
                have a 'class' key with transformer class. All other keys are
                passed into 'class' value constructor:

                    classes = [
                        {'class': StandardScaler, 'with_mean': False},
                        {'class': LabelBinarizer}
                    }]

                If None value selected, then each feature left as is.

    """
    if classes is None:
        return [(column, None) for column in columns]

    feature_defs = []

    for column in columns:
        feature_transformers = []

        for definition in classes:
            if isinstance(definition, dict):
                params = definition.copy()
                klass = params.pop('class')
                feature_transformers.append(klass(**params))
            elif isinstance(definition, type):
                feature_transformers.append(definition())

        if not feature_transformers:
            feature_transformers = None

        feature_defs.append((column, feature_transformers))

    return feature_defs
