__version__ = '1.8.1'

from .dataframe_mapper import DataFrameMapper  # NOQA
from .cross_validation import cross_val_score, GridSearchCV, RandomizedSearchCV  # NOQA
from .transformers import CategoricalImputer, FunctionTransformer  # NOQA
from .features_generator import gen_features  # NOQA
