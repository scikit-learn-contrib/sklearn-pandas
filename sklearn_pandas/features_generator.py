def gen_features(columns, classes=None, prefix='', suffix=''):
    """Generates a feature definition list which can be passed
    into DataFrameMapper

    Params:

    columns     a list of column names to generate features for.

    classes     a list of classes for each feature, a list of dictionaries with
                transformer class and init parameters, or None.

                If list of classes is provided, then each of them is
                instantiated with default arguments. Example:

                    classes = [StandardScaler, LabelBinarizer]

                If list of dictionaries is provided, then each of them should
                have a 'class' key with transformer class. All other keys are
                passed into 'class' value constructor. Example:

                    classes = [
                        {'class': StandardScaler, 'with_mean': False},
                        {'class': LabelBinarizer}
                    }]

                If None value selected, then each feature left as is.

    prefix      add prefix to transformed column names

    suffix      add suffix to transformed column names.

    """
    if classes is None:
        return [(column, None) for column in columns]

    feature_defs = []

    for column in columns:
        feature_transformers = []

        arguments = {}
        if prefix and prefix != "":
            arguments['prefix'] = prefix
        if suffix and suffix != "":
            arguments['suffix'] = suffix

        classes = [cls for cls in classes if cls is not None]
        if not classes:
            feature_defs.append((column, None, arguments))

        else:
            for definition in classes:
                if isinstance(definition, dict):
                    params = definition.copy()
                    klass = params.pop('class')
                    feature_transformers.append(klass(**params))
                else:
                    feature_transformers.append(definition())

            if not feature_transformers:
                feature_transformers = None

            feature_defs.append((column, feature_transformers, arguments))

    return feature_defs
