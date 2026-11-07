"""Shared hierarchy mappings and the registry of model types.

Moved verbatim from views.py; views.py re-exports these names so the
existing `from apps.model_training.views import ...` in
apps/demand_prediction/views.py keeps working.
"""
from django.utils.translation import gettext

CATEGORY_FIELD_MAP = {
    'category_l1': 'product_data__product_category_level1',
    'category_l2': 'product_data__product_category_level2',
    'category_l3': 'product_data__product_category_level3',
    'category_l4': 'product_data__product_category_level4',
    'category_l5': 'product_data__product_category_level5',
}


HIERARCHY_LABEL_MAP = {
    'sku': gettext('SKU级别'),
    'category_l1': gettext('一级分类'),
    'category_l2': gettext('二级分类'),
    'category_l3': gettext('三级分类'),
    'category_l4': gettext('四级分类'),
    'category_l5': gettext('五级分类'),
}


LEVEL_FIELD_MAP = {
    'category_l1': 'product_category_level1',
    'category_l2': 'product_category_level2',
    'category_l3': 'product_category_level3',
    'category_l4': 'product_category_level4',
    'category_l5': 'product_category_level5',
}


VALID_MODEL_TYPES = [
    'holt-winters', 'arima', 'arimax', 'xgboost',
    'moving_avg', 'weighted_avg', 'exp_smooth', 'seasonal_naive',
    'prophet', 'random_forest', 'croston', 'sba',
    'lstm', 'deepar'
]
