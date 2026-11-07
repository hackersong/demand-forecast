"""Training orchestration: build features, fit, score and persist models.

Moved verbatim from views.py. The HTTP layer (train_model / model_results)
stays in views.py; this module owns everything that turns a region x item
combination into ForecastModel/ForecastResult rows.
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from django.utils.translation import gettext as _

from apps.data_management.models import SalesData

from .algorithms import (
    _compute_trend_forecast,
    _croston_predict,
    _deepar_predict,
    _get_fallback_model_types,
    _get_retry_params,
    _lstm_predict,
    _sba_predict,
    _seasonal_decompose,
)
from .constants import CATEGORY_FIELD_MAP, VALID_MODEL_TYPES
from .metrics import _eval_metrics
from ..models import ForecastModel, ForecastResult

logger = logging.getLogger(__name__)


def _preprocess_time_series(monthly_series, region=None, sku=None):
    """
    月度时序数据清洗与预处理，减少数据质量问题对模型训练的影响。

    处理步骤（按顺序）：
    1. 空值检测  → 线性插值填补（兜底：中位数填充）
    2. 异常值检测 → IQR 方法识别离群点，用相邻值均值替换
    3. 极端值平滑 → Winsorize 截断（3倍IQR），避免过大/过小值扭曲模型
    4. 零值占比分析 → 记录日志但保持原样（间歇性需求留给 Croston/SBA 处理）

    参数:
        monthly_series: pd.Series，按月汇总的销售数据
        region: 可选，用于日志标识
        sku: 可选，用于日志标识

    返回:
        (clean_series: pd.Series, cleaning_log: dict)
    """
    series = monthly_series.copy()
    tag = f'[{region}/{sku}]' if region and sku else ''
    cleaning_log = {
        'original_length': len(series),
        'null_count': 0,
        'outlier_count': 0,
        'extreme_cap_count': 0,
        'original_mean': float(series.mean()) if len(series) > 0 else 0.0,
        'actions': [],
    }

    if len(series) < 3:
        cleaning_log['actions'].append('数据量不足3个月，跳过清洗')
        return series, cleaning_log

    # ═══════════════════════════════════════════════════════
    #  第1步：空值检测与插值
    # ═══════════════════════════════════════════════════════
    null_mask = series.isna()
    cleaning_log['null_count'] = int(null_mask.sum())

    if null_mask.any():
        # 使用线性插值（利用时间趋势填补缺失月份）
        series = series.interpolate(method='linear', limit_direction='both')
        # 如果首尾仍有 NaN（如起始月份缺失），用整体中位数兜底
        if series.isna().any():
            fill_value = series.median() if not series.dropna().empty else 0.0
            series = series.fillna(fill_value)
        cleaning_log['actions'].append(
            f'填补了 {cleaning_log["null_count"]} 个空值（线性插值，兜底中位数）'
        )
        logger.info(f'{tag} 填补 {cleaning_log["null_count"]} 个空值')

    # ═══════════════════════════════════════════════════════
    #  第2步：异常值检测（IQR 方法，仅对非零值）
    # ═══════════════════════════════════════════════════════
    values = series.values.astype(float)
    nonzero_mask = values > 0
    nonzero_count = int(nonzero_mask.sum())

    if nonzero_count >= 3:
        nonzero_vals = values[nonzero_mask]
        Q1 = float(np.percentile(nonzero_vals, 25))
        Q3 = float(np.percentile(nonzero_vals, 75))
        IQR = Q3 - Q1

        if IQR > 0:
            # 时序数据用 2.5×IQR（比标准 1.5×IQR 更宽松，避免误杀季节性波动）
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR

            outlier_mask = (values > upper_bound) | ((values < lower_bound) & (values > 0))
            outlier_indices = np.where(outlier_mask)[0]

            if len(outlier_indices) > 0:
                cleaning_log['outlier_count'] = int(len(outlier_indices))

                for idx in outlier_indices:
                    # 用相邻月份的均值替换（保留局部趋势信息）
                    neighbors = []
                    if idx > 0:
                        neighbors.append(values[idx - 1])
                    if idx < len(values) - 1:
                        neighbors.append(values[idx + 1])
                    replacement = float(np.mean(neighbors)) if neighbors else float(np.median(nonzero_vals))
                    values[idx] = replacement
                    series.iloc[idx] = replacement

                cleaning_log['actions'].append(
                    f'处理了 {cleaning_log["outlier_count"]} 个异常值 '
                    f'(IQR边界: [{lower_bound:.0f}, {upper_bound:.0f}])'
                )
                logger.warning(
                    f'{tag} 检测到 {cleaning_log["outlier_count"]} 个异常值 '
                    f'(边界: [{lower_bound:.0f}, {upper_bound:.0f}])'
                )

    # ═══════════════════════════════════════════════════════
    #  第3步：极端值平滑（Winsorize，3×IQR 截断）
    # ═══════════════════════════════════════════════════════
    values = series.values.astype(float)
    nonzero_vals = values[values > 0]

    if len(nonzero_vals) >= 3:
        Q1 = float(np.percentile(nonzero_vals, 25))
        Q3 = float(np.percentile(nonzero_vals, 75))
        IQR = Q3 - Q1

        if IQR > 0:
            cap_upper = Q3 + 3.0 * IQR
            cap_lower = max(0.0, Q1 - 3.0 * IQR)

            cap_mask = (values > cap_upper) | ((values < cap_lower) & (values > 0))
            cap_count = int(cap_mask.sum())

            if cap_count > 0:
                cleaning_log['extreme_cap_count'] = cap_count
                values = np.clip(values, cap_lower, cap_upper)
                series = pd.Series(values, index=series.index)
                cleaning_log['actions'].append(
                    f'Winsorize 截断 {cap_count} 个极端值'
                    f'（上限:{cap_upper:.0f}, 下限:{cap_lower:.0f}）'
                )
                logger.info(f'{tag} Winsorize 截断 {cap_count} 个极端值')

    # ═══════════════════════════════════════════════════════
    #  第4步：零值模式分析（仅记录，不修改）
    # ═══════════════════════════════════════════════════════
    zero_ratio = float((series.values == 0).sum()) / len(series) if len(series) > 0 else 0.0
    if zero_ratio > 0.5:
        cleaning_log['actions'].append(
            f'零值占比 {zero_ratio*100:.1f}%，属间歇性需求模式，保留原始零值'
        )

    # ═══════════════════════════════════════════════════════
    #  第5步：季节性诊断（仅记录，不修改数据）
    # ═══════════════════════════════════════════════════════
    seasonal_info = None
    if len(series) >= 12:
        try:
            seasonal_info = _seasonal_decompose(series)
            cleaning_log['seasonal_strength'] = round(seasonal_info['seasonal_strength'], 3)
            cleaning_log['seasonal_index'] = [round(float(x), 3) for x in seasonal_info['seasonal_index']]

            # 诊断季节性峰值月份
            peak_month_idx = int(np.argmax(seasonal_info['seasonal_index']))
            trough_month_idx = int(np.argmin(seasonal_info['seasonal_index']))
            cleaning_log['peak_month'] = peak_month_idx + 1
            cleaning_log['trough_month'] = trough_month_idx + 1

            if seasonal_info['seasonal_strength'] > 0.3:
                cleaning_log['actions'].append(
                    f'强季节性 (强度={seasonal_info["seasonal_strength"]:.2f}, '
                    f'峰值月={peak_month_idx+1}, 低谷月={trough_month_idx+1})'
                )
        except Exception:
            pass

    cleaning_log['final_mean'] = float(series.mean())
    cleaning_log['final_length'] = len(series)
    cleaning_log['zero_ratio'] = round(zero_ratio, 3)
    cleaning_log['deseasonalized_values'] = (
        [round(float(x), 2) for x in seasonal_info['deseasonalized']]
        if seasonal_info else None
    )

    # 汇总日志
    if cleaning_log['actions']:
        logger.info(f'{tag} 数据清洗完成：{"；".join(cleaning_log["actions"])}')

    return series, cleaning_log


def train_single_combination(region, sku, model_type='holt-winters', time_range=12,
                             hierarchy_level='sku', category_code=None,
                             start_date_str=None, end_date_str=None):
    """
    训练单个 region + sku/分类 组合的预测模型。
    支持 SKU 级别和产品五级分类级别的聚合训练。

    参数:
        start_date_str / end_date_str: 'YYYY-MM' 格式，指定训练数据的时间范围；
                                       若提供则优先于 time_range。
    返回: (success: bool, error_msg: str | None)
    """
    from datetime import datetime
    import calendar as cal_mod

    if model_type not in VALID_MODEL_TYPES:
        model_type = 'holt-winters'

    # ── 数据查询：SKU 级别 vs 分类级别 ──
    if hierarchy_level == 'sku':
        sales_data = SalesData.objects.filter(
            sales_region_code=region,
            product_data__material_number=sku
        ).order_by('date')
    else:
        field_name = CATEGORY_FIELD_MAP.get(hierarchy_level, 'product_data__product_category_level1')
        sales_data = SalesData.objects.filter(
            sales_region_code=region,
            **{field_name: sku}
        ).order_by('date')

    if not sales_data.exists():
        if hierarchy_level == 'sku':
            return False, f'未找到区域 {region} 和 SKU {sku} 的销售数据'
        else:
            return False, f'未找到区域 {region} 和分类 {sku} 的销售数据'

    # ── 时间范围过滤 ──
    if start_date_str and end_date_str:
        start_dt = datetime.strptime(start_date_str, '%Y-%m')
        end_dt = datetime.strptime(end_date_str, '%Y-%m')
        last_day = cal_mod.monthrange(end_dt.year, end_dt.month)[1]
        end_full = datetime(end_dt.year, end_dt.month, last_day)
        filtered_data = sales_data.filter(date__gte=start_dt, date__lte=end_full)
    else:
        end_date = sales_data.latest('date').date
        start_date = end_date - timedelta(days=30 * time_range)
        filtered_data = sales_data.filter(date__gte=start_date)

    if filtered_data.count() < 12:
        return False, f'区域 {region} SKU {sku} 的历史数据不足（至少需要12个月），当前只有{filtered_data.count()}条'

    df = pd.DataFrame(list(filtered_data.values('date', 'sales_quantity')))
    if df.empty:
        return False, f'区域 {region} SKU {sku} 的历史数据为空'
    # 分类层级下同日期可能有多条记录 → 先 groupby 求和再 reindex
    df = df.groupby('date', as_index=False)['sales_quantity'].sum()
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    df = df.asfreq('D')
    df['sales_quantity'] = df['sales_quantity'].fillna(0)
    monthly_data = df['sales_quantity'].resample('ME').sum()

    # ── 数据清洗与预处理 ──
    monthly_data, cleaning_log = _preprocess_time_series(monthly_data, region, sku)

    if len(monthly_data) < 12:
        return False, f'区域 {region} SKU {sku} 的月度数据不足（至少需要12个月），当前只有{len(monthly_data)}个月'

    if len(monthly_data) > 12:
        train_data = monthly_data[:-3]
        test_data = monthly_data[-3:]
    else:
        train_data = monthly_data
        test_data = None

    forecast_values = []
    mae = 0.0
    rmse = 0.0
    accuracy = 90.0
    test_forecast_values = []

    # ── 过拟合重训循环：最多3次尝试，逐次降参数复杂度 ──
    MAX_RETRIES = 3
    avg_monthly = float(np.mean(np.asarray(monthly_data.values, dtype=float)))
    is_overfit = True
    last_error = None

    for attempt in range(MAX_RETRIES):
        retry_params = _get_retry_params(model_type, attempt)

        forecast_values = []
        mae = 0.0
        rmse = 0.0
        accuracy = 90.0
        test_forecast_values = []

        train_error = None  # 本次训练的异常（非导入错误）

        if model_type == 'arima':
            try:
                from statsmodels.tsa.arima.model import ARIMA
            except ImportError:
                return False, 'ARIMA 模型依赖 statsmodels，请先安装该包。'
            try:
                arima_order = retry_params.get('arima_order', (1, 1, 1))
                arima_model = ARIMA(train_data, order=arima_order).fit()
                if test_data is not None:
                    test_values = test_data.astype(float).values
                    test_forecast = arima_model.forecast(steps=3)
                    test_forecast_values = [max(0.0, float(x)) for x in test_forecast.values]
                    fc_arr = np.maximum(0.0, test_forecast.values.astype(float))
                    mae = float(np.mean(np.abs(test_values - fc_arr)))
                    rmse = float(np.sqrt(np.mean((test_values - fc_arr) ** 2)))
                    # 用 actual 做分母，避免 forecast→0 时误差爆炸
                    pct_errs = np.where(test_values > 0, np.abs(test_values - fc_arr) / test_values, 0.0)
                    pct_errs = np.where((test_values == 0) & (fc_arr > 0), 1.0, pct_errs)
                    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
                forecast_values = [max(0.0, float(x)) for x in arima_model.forecast(steps=12)]
            except Exception as e:
                train_error = f'ARIMA 模型训练失败：{e}'

        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
            except ImportError:
                return False, 'XGBoost 模型依赖 xgboost，请先安装该包。'
            try:
                monthly_df = monthly_data.reset_index()
                monthly_df['month_index'] = np.arange(len(monthly_df), dtype=float)
                monthly_df['month_of_year'] = monthly_df['date'].dt.month.astype(float)
                X_all = monthly_df[['month_index', 'month_of_year']].values
                y_all = monthly_df['sales_quantity'].astype(float).values
                if len(monthly_df) > 12:
                    X_train = X_all[:-3]
                    y_train = y_all[:-3]
                    X_test = X_all[-3:]
                    y_test = y_all[-3:]
                else:
                    X_train = X_all
                    y_train = y_all
                    X_test = None
                xgb_model = XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=retry_params.get('n_estimators', 100),
                    max_depth=retry_params.get('max_depth', 6),
                    learning_rate=retry_params.get('learning_rate', 0.3),
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                if X_test is not None:
                    test_forecast = xgb_model.predict(X_test)
                    fc_arr = np.maximum(0.0, test_forecast)
                    test_forecast_values = [float(x) for x in fc_arr]
                    y_test_values = np.asarray(y_test, dtype=float)
                    mae = float(np.mean(np.abs(y_test_values - fc_arr)))
                    rmse = float(np.sqrt(np.mean((y_test_values - fc_arr) ** 2)))
                    pct_errs = np.where(y_test_values > 0, np.abs(y_test_values - fc_arr) / y_test_values, 0.0)
                    pct_errs = np.where((y_test_values == 0) & (fc_arr > 0), 1.0, pct_errs)
                    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
                future_index = np.arange(len(monthly_df), len(monthly_df) + 12, dtype=float)
                future_months = np.array([(monthly_df['date'].iloc[-1].month + i) % 12 + 1
                                          for i in range(1, 13)], dtype=float)
                future_X = np.column_stack([future_index, future_months])
                forecast_values = [max(0.0, float(x)) for x in xgb_model.predict(future_X)]
            except Exception as e:
                train_error = f'XGBoost 模型训练失败：{e}'

        elif model_type == 'moving_avg':
            # 简单移动平均 —— 稳定的基线模型，无需重训降参
            window = min(6, max(2, len(monthly_data) // 4))
            monthly_arr = monthly_data.values.astype(float)
            if test_data is not None:
                test_forecast_values = [max(0.0, float(np.mean(monthly_arr[-window:]))) for _ in range(3)]
                fc_arr = np.array(test_forecast_values)
                test_act_arr = np.asarray(test_data.values, dtype=float)
                mae = float(np.mean(np.abs(test_act_arr - fc_arr)))
                rmse = float(np.sqrt(np.mean((test_act_arr - fc_arr) ** 2)))
                pct_errs = np.where(test_act_arr > 0, np.abs(test_act_arr - fc_arr) / test_act_arr, 0.0)
                pct_errs = np.where((test_act_arr == 0) & (fc_arr > 0), 1.0, pct_errs)
                accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
            forecast_values = _compute_trend_forecast(monthly_arr, horizon=12)

        elif model_type == 'weighted_avg':
            # 加权移动平均 —— 近期数据权重更大，无需重训降参
            window = min(6, max(2, len(monthly_data) // 3))
            monthly_arr = monthly_data.values.astype(float)
            recent_vals = monthly_arr[-window:]
            w = np.arange(1, len(recent_vals) + 1, dtype=float)
            w = w / w.sum()
            wma_val = float(np.sum(recent_vals * w))
            if test_data is not None:
                test_forecast_values = [max(0.0, wma_val) for _ in range(3)]
                fc_arr = np.array(test_forecast_values)
                test_act_arr = np.asarray(test_data.values, dtype=float)
                mae = float(np.mean(np.abs(test_act_arr - fc_arr)))
                rmse = float(np.sqrt(np.mean((test_act_arr - fc_arr) ** 2)))
                pct_errs = np.where(test_act_arr > 0, np.abs(test_act_arr - fc_arr) / test_act_arr, 0.0)
                pct_errs = np.where((test_act_arr == 0) & (fc_arr > 0), 1.0, pct_errs)
                accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
            forecast_values = _compute_trend_forecast(monthly_arr, horizon=12)

        elif model_type == 'exp_smooth':
            # 指数平滑（Holt-Winters）—— 处理趋势+季节性
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
            except ImportError:
                return False, 'exp_smooth 依赖 statsmodels，请先安装该包。'
            try:
                train_vals_arr = train_data.astype(float).values
                n_train = len(train_vals_arr)
                use_seasonal = retry_params.get('seasonal_enabled', len(train_data) >= 24)
                seasonal_periods = retry_params.get('seasonal_periods', 12)

                # 季节性周期安全校验（与 holt-winters 一致）
                if use_seasonal and seasonal_periods is not None:
                    if seasonal_periods > n_train:
                        for sp in [12, 6, 4, 3]:
                            if sp <= n_train:
                                seasonal_periods = sp
                                break
                        else:
                            seasonal_periods = None

                    if seasonal_periods is not None and n_train >= seasonal_periods:
                        es_model = ExponentialSmoothing(
                            train_vals_arr, trend='add', seasonal='add', seasonal_periods=seasonal_periods
                        ).fit()
                    else:
                        es_model = ExponentialSmoothing(
                            train_vals_arr, trend='add', seasonal=None
                        ).fit()
                else:
                    es_model = ExponentialSmoothing(
                        train_vals_arr, trend='add', seasonal=None
                    ).fit()
                if test_data is not None:
                    test_vals = test_data.astype(float).values
                    test_forecast = es_model.forecast(steps=3)
                    test_forecast_values = [max(0.0, float(x)) for x in test_forecast]
                    fc_arr = np.maximum(0.0, np.asarray(test_forecast, dtype=float))
                    mae = float(np.mean(np.abs(test_vals - fc_arr)))
                    rmse = float(np.sqrt(np.mean((test_vals - fc_arr) ** 2)))
                    pct_errs = np.where(test_vals > 0, np.abs(test_vals - fc_arr) / test_vals, 0.0)
                    pct_errs = np.where((test_vals == 0) & (fc_arr > 0), 1.0, pct_errs)
                    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
                forecast_values = [max(0.0, float(x)) for x in es_model.forecast(steps=12)]
            except Exception as e:
                train_error = f'exp_smooth 训练失败：{e}'

        elif model_type == 'seasonal_naive':
            # 季节性朴素预测 —— 无需重训降参
            # 需要至少12个月数据，用最近12个月作为未来12个月的预测
            monthly_arr = monthly_data.values.astype(float)
            if len(monthly_data) >= 12:
                seasonal_vals = monthly_arr[-12:]
                if test_data is not None:
                    train_arr = train_data.values.astype(float)
                    if len(train_arr) >= 3:
                        offset = len(train_arr) - 3
                        test_forecast_values = [max(0.0, float(train_arr[offset + i])) for i in range(3)]
                    else:
                        test_forecast_values = [max(0.0, float(np.mean(train_arr))) for _ in range(3)]
                    fc_arr = np.array(test_forecast_values)
                    test_act_arr = np.asarray(test_data.values, dtype=float)
                    mae = float(np.mean(np.abs(test_act_arr - fc_arr)))
                    rmse = float(np.sqrt(np.mean((test_act_arr - fc_arr) ** 2)))
                    pct_errs = np.where(test_act_arr > 0, np.abs(test_act_arr - fc_arr) / test_act_arr, 0.0)
                    pct_errs = np.where((test_act_arr == 0) & (fc_arr > 0), 1.0, pct_errs)
                    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
                forecast_values = [max(0.0, float(v)) for v in seasonal_vals]
            else:
                mean_val = float(np.mean(monthly_arr))
                forecast_values = _compute_trend_forecast(monthly_arr, horizon=12)
                if test_data is not None:
                    test_forecast_values = [max(0.0, mean_val) for _ in range(3)]
                    fc_arr = np.array(test_forecast_values)
                    test_act_arr = np.asarray(test_data.values, dtype=float)
                    mae = float(np.mean(np.abs(test_act_arr - fc_arr)))
                    rmse = float(np.sqrt(np.mean((test_act_arr - fc_arr) ** 2)))
                    pct_errs = np.where(test_act_arr > 0, np.abs(test_act_arr - fc_arr) / test_act_arr, 0.0)
                    pct_errs = np.where((test_act_arr == 0) & (fc_arr > 0), 1.0, pct_errs)
                    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0

        elif model_type == 'arimax':
            try:
                from statsmodels.tsa.arima.model import ARIMA
            except ImportError:
                return False, 'ARIMAX 依赖 statsmodels，请先安装该包。'
            try:
                train_arr = train_data.values.astype(float)
                n_train = len(train_arr)
                exog_train = np.arange(n_train, dtype=float).reshape(-1, 1)
                arimax_order = retry_params.get('arimax_order', (1, 1, 1))
                model = ARIMA(train_arr, exog=exog_train, order=arimax_order).fit()
                if test_data is not None:
                    test_values = test_data.astype(float).values
                    exog_test = np.arange(n_train, n_train + 3, dtype=float).reshape(-1, 1)
                    test_fc = model.forecast(steps=3, exog=exog_test)
                    test_forecast_values = [max(0.0, float(x)) for x in test_fc]
                    mae, rmse, accuracy = _eval_metrics(test_values, test_forecast_values)
                exog_future = np.arange(n_train, n_train + 12, dtype=float).reshape(-1, 1)
                forecast_values = [max(0.0, float(x)) for x in model.forecast(steps=12, exog=exog_future)]
            except Exception as e:
                train_error = f'ARIMAX 训练失败：{e}'

        elif model_type == 'prophet':
            try:
                from prophet import Prophet
            except ImportError:
                return False, 'Prophet 依赖 prophet 包，请先安装。'
            try:
                prophet_df = pd.DataFrame({'ds': monthly_data.index, 'y': monthly_data.values.astype(float)})
                y_seasonality = retry_params.get('yearly_seasonality', len(monthly_data) >= 18)
                cp_scale = retry_params.get('changepoint_prior_scale', 0.05)
                model = Prophet(
                    weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=y_seasonality,
                    changepoint_prior_scale=cp_scale
                )
                model.fit(prophet_df)
                if test_data is not None:
                    test_fc_df = model.make_future_dataframe(periods=3, freq='ME')
                    test_fc = model.predict(test_fc_df).tail(3)['yhat'].values
                    test_forecast_values = [max(0.0, float(x)) for x in test_fc]
                    mae, rmse, accuracy = _eval_metrics(test_data.values.astype(float), test_forecast_values)
                future = model.make_future_dataframe(periods=12, freq='ME')
                forecast_values = [max(0.0, float(x)) for x in model.predict(future).tail(12)['yhat'].values]
            except Exception as e:
                train_error = f'Prophet 训练失败：{e}'

        elif model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestRegressor
            except ImportError:
                return False, 'Random Forest 依赖 scikit-learn，请先安装。'
            try:
                monthly_df = monthly_data.reset_index()
                monthly_df['month_index'] = np.arange(len(monthly_df), dtype=float)
                monthly_df['month_of_year'] = monthly_df['date'].dt.month.astype(float)
                X_all = monthly_df[['month_index', 'month_of_year']].values
                y_all = monthly_df['sales_quantity'].astype(float).values
                if len(monthly_df) > 12:
                    X_tr, X_te = X_all[:-3], X_all[-3:]
                    y_tr, y_te = y_all[:-3], y_all[-3:]
                else:
                    X_tr, X_te, y_tr, y_te = X_all, None, y_all, None
                rf_n = retry_params.get('n_estimators', 150)
                rf_depth = retry_params.get('max_depth', None)
                model = RandomForestRegressor(
                    n_estimators=rf_n, max_depth=rf_depth, random_state=42, n_jobs=-1
                )
                model.fit(X_tr, y_tr)
                if X_te is not None:
                    test_fc = model.predict(X_te)
                    test_forecast_values = [max(0.0, float(x)) for x in test_fc]
                    mae, rmse, accuracy = _eval_metrics(y_te, test_forecast_values)
                last_idx = len(monthly_df)
                future_idx = np.arange(last_idx, last_idx + 12, dtype=float)
                future_months = np.array([(monthly_df['date'].iloc[-1].month + i) % 12 + 1
                                          for i in range(1, 13)], dtype=float)
                future_X = np.column_stack([future_idx, future_months])
                forecast_values = [max(0.0, float(x)) for x in model.predict(future_X)]
            except Exception as e:
                train_error = f'Random Forest 训练失败：{e}'

        elif model_type == 'croston':
            try:
                forecast_values = [max(0.0, float(x)) for x in _croston_predict(monthly_data, periods=12)]
                if test_data is not None:
                    test_forecast_values = [max(0.0, float(x)) for x in _croston_predict(train_data, periods=3)]
                    mae, rmse, accuracy = _eval_metrics(test_data.values.astype(float), test_forecast_values)
            except Exception as e:
                train_error = f'Croston 训练失败：{e}'

        elif model_type == 'sba':
            try:
                forecast_values = [max(0.0, float(x)) for x in _sba_predict(monthly_data, periods=12)]
                if test_data is not None:
                    test_forecast_values = [max(0.0, float(x)) for x in _sba_predict(train_data, periods=3)]
                    mae, rmse, accuracy = _eval_metrics(test_data.values.astype(float), test_forecast_values)
            except Exception as e:
                train_error = f'SBA 训练失败：{e}'

        elif model_type == 'lstm':
            try:
                import tensorflow as _tf_lstm
            except ImportError:
                return False, 'LSTM 依赖 TensorFlow，请先安装。'
            try:
                if len(monthly_data) < 12:
                    return False, 'LSTM 至少需要12个月数据'
                lstm_epochs = retry_params.get('epochs', 60)
                lstm_lookback = retry_params.get('lookback', 6)
                forecast_values = [max(0.0, float(x)) for x in _lstm_predict(monthly_data, periods=12, epochs=lstm_epochs, lookback=lstm_lookback)]
                if test_data is not None:
                    test_forecast_values = [max(0.0, float(x)) for x in _lstm_predict(train_data, periods=3, epochs=lstm_epochs, lookback=lstm_lookback)]
                    mae, rmse, accuracy = _eval_metrics(test_data.values.astype(float), test_forecast_values)
            except Exception as e:
                train_error = f'LSTM 训练失败：{e}'

        elif model_type == 'deepar':
            try:
                import tensorflow as _tf_deepar
            except ImportError:
                return False, 'DeepAR 依赖 TensorFlow，请先安装。'
            try:
                if len(monthly_data) < 12:
                    return False, 'DeepAR 至少需要12个月数据'
                deepar_epochs = retry_params.get('epochs', 50)
                deepar_lookback = retry_params.get('lookback', 6)
                forecast_values = [max(0.0, float(x)) for x in _deepar_predict(monthly_data, periods=12, epochs=deepar_epochs, lookback=deepar_lookback)]
                if test_data is not None:
                    test_forecast_values = [max(0.0, float(x)) for x in _deepar_predict(train_data, periods=3, epochs=deepar_epochs, lookback=deepar_lookback)]
                    mae, rmse, accuracy = _eval_metrics(test_data.values.astype(float), test_forecast_values)
            except Exception as e:
                train_error = f'DeepAR 训练失败：{e}'

        elif model_type == 'holt-winters':
            # 教科书 Holt-Winters 指数平滑（additive trend + additive seasonality）
            # 当数据不足24个月时自动退化为 Holt 线性趋势（无季节项）
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
            except ImportError:
                # statsmodels 不可用时回退到线性回归+季节分解
                train_error = None  # 不使用重训循环，直接走 fallback 逻辑
                monthly_df = monthly_data.reset_index()
                monthly_df['month_index'] = np.arange(len(monthly_df), dtype=float)
                X_all = np.asarray(monthly_df['month_index'].values, dtype=float)
                y_all = np.asarray(monthly_df['sales_quantity'].values, dtype=float)
                slope, intercept = np.polyfit(X_all, y_all, 1)
                trend_line = intercept + slope * X_all
                residuals = y_all - trend_line
                seasonal_means = np.zeros(12)
                seasonal_counts = np.zeros(12)
                for idx, row in monthly_df.iterrows():
                    m = row['date'].month - 1
                    seasonal_means[m] += residuals[idx]
                    seasonal_counts[m] += 1
                for m in range(12):
                    if seasonal_counts[m] > 0:
                        seasonal_means[m] /= seasonal_counts[m]
                if test_data is not None:
                    train_slope, train_intercept = np.polyfit(X_all[:-3], y_all[:-3], 1)
                    test_trend = train_intercept + train_slope * np.arange(len(y_all)-3, len(y_all), dtype=float)
                    test_fc = test_trend + np.array([seasonal_means[(monthly_df['date'].iloc[len(y_all)-3+j].month-1)] for j in range(3)])
                    mae, rmse, accuracy = _eval_metrics(y_all[-3:], np.maximum(0.0, test_fc))
                    test_forecast_values = [float(v) for v in np.maximum(0.0, test_fc)]
                forecast_raw = intercept + slope * np.arange(len(y_all), len(y_all)+12, dtype=float)
                last_d = monthly_df['date'].iloc[-1]
                for step in range(1, 13):
                    fd = last_d + pd.DateOffset(months=step)
                    forecast_raw[step-1] += seasonal_means[fd.month - 1]
                forecast_values = [max(0.0, float(v)) for v in forecast_raw]
            else:
                try:
                    train_vals_arr = train_data.astype(float).values
                    n_train = len(train_vals_arr)
                    use_seasonal = retry_params.get('use_seasonal', n_train >= 12)
                    seasonal_periods = retry_params.get('seasonal_periods', 12)

                    # 季节性周期安全校验：若重试参数指定的周期超过数据量，自动降级
                    # 不从 [12,6,4,3] 重新选择，避免覆盖重试机制的降参意图
                    if use_seasonal and seasonal_periods is not None:
                        if seasonal_periods > n_train:
                            # 周期过大，降级到数据支持的最大合理周期
                            for sp in [12, 6, 4, 3]:
                                if sp <= n_train:
                                    seasonal_periods = sp
                                    break
                            else:
                                seasonal_periods = None

                        if seasonal_periods is not None and n_train >= seasonal_periods:
                            hw_model = ExponentialSmoothing(
                                train_vals_arr, trend='add', seasonal='add',
                                seasonal_periods=seasonal_periods
                            ).fit()
                        else:
                            hw_model = ExponentialSmoothing(
                                train_vals_arr, trend='add', seasonal=None
                            ).fit()
                    else:
                        hw_model = ExponentialSmoothing(
                            train_vals_arr, trend='add', seasonal=None
                        ).fit()
                    if test_data is not None:
                        test_vals = test_data.astype(float).values
                        test_forecast = hw_model.forecast(steps=3)
                        test_forecast_values = [max(0.0, float(x)) for x in test_forecast]
                        mae, rmse, accuracy = _eval_metrics(test_vals, test_forecast_values)
                    forecast_values = [max(0.0, float(x)) for x in hw_model.forecast(steps=12)]
                except Exception as e:
                    train_error = f'Holt-Winters 训练失败：{e}'

        else:
            # 未知模型类型或兜底：线性回归趋势 + 月份季节分量分解
            monthly_df = monthly_data.reset_index()
            monthly_df['month_index'] = np.arange(len(monthly_df), dtype=float)
            X = np.asarray(monthly_df['month_index'].values, dtype=float)
            y = np.asarray(monthly_df['sales_quantity'].values, dtype=float)
            n_all = len(y)

            slope, intercept = np.polyfit(X, y, 1)
            trend_line = intercept + slope * X
            residuals = y - trend_line

            seasonal_means = np.zeros(12)
            seasonal_counts = np.zeros(12)
            for idx, row in monthly_df.iterrows():
                m = row['date'].month - 1
                seasonal_means[m] += residuals[idx]
                seasonal_counts[m] += 1
            for m in range(12):
                if seasonal_counts[m] > 0:
                    seasonal_means[m] /= seasonal_counts[m]

            if test_data is not None:
                train_X = np.asarray(X[:-3], dtype=float)
                train_y = np.asarray(y[:-3], dtype=float)
                test_X = np.asarray(X[-3:], dtype=float)
                test_y = np.asarray(y[-3:], dtype=float)
                train_slope, train_intercept = np.polyfit(train_X, train_y, 1)
                train_trend = train_intercept + train_slope * train_X
                train_resid = train_y - train_trend
                train_seasonal = np.zeros(12)
                train_seasonal_cnt = np.zeros(12)
                train_dates = monthly_df['date'].iloc[:-3]
                for idx2, d in enumerate(train_dates):
                    m2 = d.month - 1
                    train_seasonal[m2] += train_resid[idx2]
                    train_seasonal_cnt[m2] += 1
                for m2 in range(12):
                    if train_seasonal_cnt[m2] > 0:
                        train_seasonal[m2] /= train_seasonal_cnt[m2]
                test_trend_pred = train_intercept + train_slope * test_X
                test_season_pred = np.array([
                    train_seasonal[(monthly_df['date'].iloc[n_all - 3 + j].month - 1)]
                    for j in range(3)
                ])
                test_forecast = test_trend_pred + test_season_pred
                fc_arr = np.maximum(0.0, test_forecast)
                test_forecast_values = [float(v) for v in fc_arr]
                mae, rmse, accuracy = _eval_metrics(test_y, fc_arr)
            else:
                mae = 0.0
                rmse = 0.0
                accuracy = 90.0

            future_months = []
            last_date = monthly_df['date'].iloc[-1]
            for step in range(1, 13):
                future_date = last_date + pd.DateOffset(months=step)
                future_months.append(future_date.month - 1)

            future_X = np.arange(n_all, n_all + 12, dtype=float)
            trend_forecast = intercept + slope * future_X
            season_forecast = np.array([seasonal_means[m] for m in future_months])
            forecast_raw = trend_forecast + season_forecast
            forecast_values = [max(0.0, float(v)) for v in forecast_raw]

        # ── 训练异常处理 ──
        if train_error is not None:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f'[{region}/{sku}/{model_type}] 第{attempt+1}次训练异常: {train_error}，尝试降参重训')
                continue
            else:
                return False, f'{model_type} 训练失败（重试{MAX_RETRIES}次后仍报错）：{train_error}'

        # ── 过拟合判断 ──
        if test_data is not None and mae > 0:
            is_overfit = mae > avg_monthly * 0.3
        else:
            # 没有测试集或MAE为0，视为未过拟合
            is_overfit = False

        if not is_overfit:
            if attempt > 0:
                logger.info(f'[{region}/{sku}/{model_type}] 第{attempt+1}次训练成功，已消除过拟合 (MAE={mae:.0f}, 阈值={avg_monthly*0.3:.0f})')
            break  # 模型质量合格，跳出循环
        else:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f'[{region}/{sku}/{model_type}] 第{attempt+1}次过拟合 (MAE={mae:.0f} > 30%月均={avg_monthly*0.3:.0f})，降参重训')
            else:
                logger.warning(f'[{region}/{sku}/{model_type}] 已降参{MAX_RETRIES}次均过拟合，放弃该模型 (MAE={mae:.0f})')

    # 过拟合模型仍保存（标记 is_overfit=True），避免因所有模型均过拟合导致训练无结果
    # 后续在 train_model 中会优先选非过拟合模型，全部过拟合时选准确率最高的
    
    # ── 保存模型（使用循环中最后一次/最佳结果） ──
    model_name = f"{region}_{sku}_{model_type}"

    # 计算 display_name：SKU 级别取物料名称，分类级别取 sku 本身
    display_name = sku
    if hierarchy_level == 'sku':
        from apps.data_management.models import ProductData as PD
        pd_obj = PD.objects.filter(material_number=sku).first()
        if pd_obj and pd_obj.material_name:
            display_name = pd_obj.material_name

    forecast_model, created = ForecastModel.objects.update_or_create(
        sales_region_code=region,
        material_number=sku,
        model_type=model_type,
        hierarchy_level=hierarchy_level,
        category_code=category_code if hierarchy_level != 'sku' else None,
        defaults={
            'model_name': model_name,
            'display_name': display_name,
            'training_start_date': monthly_data.index[0].date(),
            'training_end_date': monthly_data.index[-1].date(),
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'is_overfit': is_overfit
        }
    )

    ForecastResult.objects.filter(
        sales_region_code=region,
        material_number=sku,
        hierarchy_level=hierarchy_level,
        category_code=category_code if hierarchy_level != 'sku' else None,
        model_used=model_type
    ).delete()

    # --- 获取全部历史销售数据（不限时间范围），按月汇总 ---
    if hierarchy_level == 'sku':
        all_sales = SalesData.objects.filter(
            sales_region_code=region,
            product_data__material_number=sku
        ).order_by('date')
    else:
        field_name = CATEGORY_FIELD_MAP.get(hierarchy_level, 'product_data__product_category_level1')
        all_sales = SalesData.objects.filter(
            sales_region_code=region,
            **{field_name: sku}
        ).order_by('date')
    all_sales_df = pd.DataFrame(list(all_sales.values('date', 'sales_quantity')))
    if not all_sales_df.empty:
        # 分类层级下同日期可能有多条记录 → 先 groupby 求和再 reindex
        all_sales_df = all_sales_df.groupby('date', as_index=False)['sales_quantity'].sum()
        all_sales_df = all_sales_df.sort_values('date')
        all_sales_df.set_index('date', inplace=True)
        all_sales_df = all_sales_df.asfreq('D')
        all_sales_df['sales_quantity'] = all_sales_df['sales_quantity'].fillna(0)
        all_monthly = all_sales_df['sales_quantity'].resample('ME').sum()
    else:
        all_monthly = monthly_data.copy()

    # ── 数据清洗与预处理（回测数据） ──
    all_monthly, bt_cleaning_log = _preprocess_time_series(all_monthly, region, sku)

    # === 扩张窗口回测（Expanding Window Backtesting） ===
    # 核心思想：对每个历史月份，仅使用该月份之前的数据训练模型并预测该月份。
    # 这是真正的"样本外"回测。
    # min_train=12：需要至少12个月训练数据才能启动回测（保证季节性模型有足够历史）
    min_train = 12
    backtest_predictions = {}     # {Pandas Timestamp: predicted_float}
    naive_predictions = {}        # 朴素基线预测（用上月值）：{Pandas Timestamp: float}
    backtest_start_idx = min_train

    all_values = np.asarray(all_monthly.values, dtype=float)
    total_months = len(all_monthly)

    if total_months <= min_train:
        # 历史数据不足，无法启动扩张窗口回测
        backtest_start_idx = total_months

    for i in range(min_train, total_months):
        # 仅使用索引 [0, i) 的数据作为训练集，预测第 i 个月
        train_window = all_values[:i]
        train_x = np.arange(i, dtype=float)

        # --- 朴素基线：用上个月的值作为预测（业界通用基线） ---
        naive_pred = float(train_window[-1]) if i > 0 else 0.0
        naive_predictions[all_monthly.index[i]] = naive_pred

        # --- 模型预测 ---
        bt_pred = None
        if model_type == 'arima':
            try:
                from statsmodels.tsa.arima.model import ARIMA as BT_ARIMA
                bt_arima = BT_ARIMA(train_window, order=(1, 1, 1)).fit()
                bt_pred = float(bt_arima.forecast(steps=1))
            except Exception:
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor as BT_XGB
                bt_months = np.array([(m % 12) + 1 for m in range(i)], dtype=float)
                bt_features = np.column_stack([train_x, bt_months])
                bt_xgb = BT_XGB(objective='reg:squarederror', n_estimators=100, random_state=42)
                bt_xgb.fit(bt_features, train_window)
                target_month = float((i % 12) + 1)
                bt_pred = float(bt_xgb.predict(np.array([[float(i), target_month]]))[0])
            except Exception:
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'moving_avg':
            # 移动平均 + 季节调校：用移动平均捕捉趋势，季节指数纠正相位
            w_sz = min(6, max(3, i // 4))
            base_pred = float(np.mean(train_window[-w_sz:]))
            # 若有≥12个月训练数据，计算季节指数进行调校
            if i >= 12:
                try:
                    si = _seasonal_decompose(pd.Series(train_window))
                    target_m = i % 12
                    season_factor = float(si['seasonal_index'][target_m])
                    # 季节性强度越高，调校幅度越大
                    blend = si['seasonal_strength']
                    bt_pred = base_pred * (1.0 - blend) + base_pred * season_factor * blend
                except Exception:
                    bt_pred = base_pred
            else:
                bt_pred = base_pred
        elif model_type == 'weighted_avg':
            # 加权移动平均 + 季节调校
            w_sz = min(6, max(3, i // 4))
            recent = train_window[-w_sz:]
            w = np.arange(1, len(recent) + 1, dtype=float)
            w = w / w.sum()
            base_pred = float(np.sum(recent * w))
            if i >= 12:
                try:
                    si = _seasonal_decompose(pd.Series(train_window))
                    target_m = i % 12
                    season_factor = float(si['seasonal_index'][target_m])
                    blend = si['seasonal_strength']
                    bt_pred = base_pred * (1.0 - blend) + base_pred * season_factor * blend
                except Exception:
                    bt_pred = base_pred
            else:
                bt_pred = base_pred
        elif model_type == 'holt-winters':
            # 教科书 Holt-Winters 回测（与训练逻辑一致：自适应季节周期）
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing as BT_ES
                # 按优先级选择适配当前窗口的最大合理周期
                for sp in [12, 6, 4, 3]:
                    if sp <= i:
                        bt_es = BT_ES(train_window, trend='add', seasonal='add',
                                      seasonal_periods=sp).fit()
                        break
                else:
                    bt_es = BT_ES(train_window, trend='add', seasonal=None).fit()
                bt_pred = float(bt_es.forecast(steps=1))
            except Exception:
                # statsmodels 不可用或拟合失败时，回退到趋势+季节分解
                s, b = np.polyfit(train_x, train_window, 1)
                trend_line = b + s * train_x
                detrended = train_window - trend_line
                seas = np.zeros(12); sc = np.zeros(12)
                for m in range(i):
                    mi = m % 12
                    seas[mi] += detrended[m]; sc[mi] += 1
                for mi in range(12):
                    if sc[mi] > 0: seas[mi] /= sc[mi]
                bt_pred = (b + s * float(i)) + seas[i % 12]
        elif model_type == 'exp_smooth':
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing as BT_ES
                for sp in [12, 6, 4, 3]:
                    if sp <= i:
                        bt_es = BT_ES(train_window, trend='add', seasonal='add',
                                      seasonal_periods=sp).fit()
                        break
                else:
                    bt_es = BT_ES(train_window, trend='add', seasonal=None).fit()
                bt_pred = float(bt_es.forecast(steps=1))
            except Exception:
                # 回退：用线性外推
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'seasonal_naive':
            # 核心：使用一年前的同月数据
            if i >= 12:
                bt_pred = float(all_values[i - 12])
            else:
                # 不足12个月时回退到朴素基线（上月值）
                bt_pred = naive_pred
        elif model_type == 'arimax':
            try:
                from statsmodels.tsa.arima.model import ARIMA as BT_ARIMAX
                exog_bt = np.arange(i, dtype=float).reshape(-1, 1)
                bt_arimax = BT_ARIMAX(train_window, exog=exog_bt, order=(1, 1, 1)).fit()
                exog_next = np.array([[float(i)]])
                bt_pred = float(bt_arimax.forecast(steps=1, exog=exog_next))
            except Exception:
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'prophet':
            try:
                from prophet import Prophet as BT_Prophet
                bt_df = pd.DataFrame({
                    'ds': pd.to_datetime(all_monthly.index[:i]),
                    'y': train_window
                })
                bt_prophet = BT_Prophet(weekly_seasonality=False, daily_seasonality=False,
                                        yearly_seasonality=i >= 18)
                bt_prophet.fit(bt_df)
                bt_future = bt_prophet.make_future_dataframe(periods=1, freq='ME')
                bt_pred = float(bt_prophet.predict(bt_future).tail(1)['yhat'].values[0])
            except Exception:
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestRegressor as BT_RF
                bt_months = np.array([(m % 12) + 1 for m in range(i)], dtype=float)
                bt_features = np.column_stack([train_x, bt_months])
                bt_rf = BT_RF(n_estimators=100, random_state=42, n_jobs=-1)
                bt_rf.fit(bt_features, train_window)
                target_month = float((i % 12) + 1)
                bt_pred = float(bt_rf.predict(np.array([[float(i), target_month]]))[0])
            except Exception:
                s, b = np.polyfit(train_x, train_window, 1)
                bt_pred = b + s * float(i)
        elif model_type == 'croston':
            bt_pred = float(_croston_predict(train_window, periods=1)[0])
        elif model_type == 'sba':
            bt_pred = float(_sba_predict(train_window, periods=1)[0])
        elif model_type == 'lstm':
            try:
                bt_pred = float(_lstm_predict(train_window, periods=1, epochs=30)[0])
            except Exception:
                bt_pred = naive_pred
        elif model_type == 'deepar':
            try:
                bt_pred = float(_deepar_predict(train_window, periods=1, epochs=50)[0])
            except Exception:
                bt_pred = naive_pred
        else:
            # holt-winters / 线性回归 + 季节分解：
            # 先用线性回归拟合趋势，再计算各月季节分量，避免纯趋势外推导致的1个月时滞
            s, b = np.polyfit(train_x, train_window, 1)
            trend_line = b + s * train_x
            residuals = train_window - trend_line

            # 计算每个月份（1~12）的季节分量均值
            seasonal_means = np.zeros(12)
            seasonal_counts = np.zeros(12)
            for m in range(i):
                mi = m % 12
                seasonal_means[mi] += residuals[m]
                seasonal_counts[mi] += 1
            # 避免除零，未出现的月份使用0
            for mi in range(12):
                if seasonal_counts[mi] > 0:
                    seasonal_means[mi] /= seasonal_counts[mi]

            trend_pred = b + s * float(i)
            season_pred = seasonal_means[i % 12]
            bt_pred = trend_pred + season_pred

        backtest_predictions[all_monthly.index[i]] = max(0.0, float(bt_pred))

    # --- 准备产品信息（SKU级别时查物料名和分类，分类级别时填充对应分类字段） ---
    pd_info = {}
    if hierarchy_level == 'sku':
        from apps.data_management.models import ProductData as PD
        pd_obj = PD.objects.filter(material_number=sku).first()
        if pd_obj:
            pd_info = {
                'material_name': pd_obj.material_name or '',
                'category_l1': pd_obj.product_category_level1 or '',
                'category_l2': pd_obj.product_category_level2 or '',
                'category_l3': pd_obj.product_category_level3 or '',
                'category_l4': pd_obj.product_category_level4 or '',
                'category_l5': pd_obj.product_category_level5 or '',
            }
    else:
        pd_info = {
            'material_name': '',
            'category_l1': '',
            'category_l2': '',
            'category_l3': '',
            'category_l4': '',
            'category_l5': '',
        }
        pd_info[hierarchy_level] = sku  # 填充对应层级的分类值

    cat_code_val = category_code if hierarchy_level != 'sku' else None

    # --- 为全部历史月份创建 ForecastResult ---
    for i in range(total_months):
        idx = all_monthly.index[i]
        actual_val = float(all_monthly.iloc[i])

        if i >= backtest_start_idx:
            pred_val = backtest_predictions.get(idx, naive_predictions.get(idx, 0.0))
        else:
            # 预热期：基于可用历史数据做线性趋势外推（一步前向预测）
            if i > 1:
                warmup_x = np.arange(i, dtype=float)
                warmup_y = all_values[:i]
                ws, wb = np.polyfit(warmup_x, warmup_y, 1)
                pred_val = float(wb + ws * float(i))
            elif i > 0:
                pred_val = float(all_values[0])
            else:
                pred_val = actual_val

        ForecastResult.objects.create(
            sales_region_code=region,
            material_number=sku,
            hierarchy_level=hierarchy_level,
            category_code=cat_code_val,
            material_name=pd_info.get('material_name', ''),
            category_l1=pd_info.get('category_l1', ''),
            category_l2=pd_info.get('category_l2', ''),
            category_l3=pd_info.get('category_l3', ''),
            category_l4=pd_info.get('category_l4', ''),
            category_l5=pd_info.get('category_l5', ''),
            date=idx.date(),
            forecast_quantity=pred_val,
            actual_quantity=actual_val,
            model_used=model_type
        )

    # --- 未来月份预测（无实际销量） ---
    last_date = all_monthly.index[-1]
    for i, value in enumerate(forecast_values, start=1):
        forecast_date = last_date + pd.DateOffset(months=i)
        ForecastResult.objects.create(
            sales_region_code=region,
            material_number=sku,
            hierarchy_level=hierarchy_level,
            category_code=cat_code_val,
            material_name=pd_info.get('material_name', ''),
            category_l1=pd_info.get('category_l1', ''),
            category_l2=pd_info.get('category_l2', ''),
            category_l3=pd_info.get('category_l3', ''),
            category_l4=pd_info.get('category_l4', ''),
            category_l5=pd_info.get('category_l5', ''),
            date=forecast_date.date(),
            forecast_quantity=value,
            model_used=model_type
        )

    # --- 基于扩张窗口回测数据重新计算 MAE/RMSE/准确率 + 朴素基线对比 ---
    paired = ForecastResult.objects.filter(
        sales_region_code=region, material_number=sku,
        hierarchy_level=hierarchy_level,
        category_code=cat_code_val,
        model_used=model_type, actual_quantity__isnull=False
    ).order_by('date')
    if paired.exists():
        results_list = list(paired)
        bt_results = results_list[backtest_start_idx:] if len(results_list) > backtest_start_idx else results_list
        all_for = np.array([float(r.forecast_quantity) for r in bt_results])
        all_act = np.array([float(r.actual_quantity) for r in bt_results])
        mae = float(np.mean(np.abs(all_for - all_act)))
        rmse = float(np.sqrt(np.mean((all_for - all_act) ** 2)))
        # 用 actual 做分母（与 _eval_metrics 保持一致），避免 forecast 极小时误差爆炸
        pct_errs = np.where(all_act > 0, np.abs(all_act - all_for) / all_act, 0.0)
        pct_errs = np.where((all_act == 0) & (all_for > 0), 1.0, pct_errs)
        accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0
        is_overfit = mae > np.mean(all_act) * 0.3 if mae > 0 else False

        # 朴素基线："用上月实际值预测当月"，MASE < 1 表示模型优于朴素基线
        naive_mae = float(np.mean([abs(all_act[i] - all_act[i-1])
                                   for i in range(1, len(all_act))])) if len(all_act) > 1 else mae
        mase = round(mae / naive_mae, 3) if naive_mae > 0 else 1.0

        forecast_model.accuracy = accuracy
        forecast_model.mae = mae
        forecast_model.rmse = rmse
        forecast_model.is_overfit = is_overfit
        forecast_model.save(update_fields=['accuracy', 'mae', 'rmse', 'is_overfit'])

        # ── 后测质量检测：准确率 < 50% 或过拟合时自动切换替代模型 ──
        if accuracy < 50.0 or is_overfit:
            logger.warning(
                f'[{region}/{sku}/{model_type}] 后测质量不达标: '
                f'准确率={accuracy:.1f}%, 过拟合={is_overfit}，触发自动重训'
            )
            fallback_types = _get_fallback_model_types(model_type)
            if not fallback_types:
                # 没有替代模型，即使质量不达标也保留当前模型（总比没有好）
                logger.warning(
                    f'[{region}/{sku}] {model_type} 质量不达标但无替代模型，保留当前模型 '
                    f'(准确率={accuracy:.1f}%)'
                )
                return True, None
            for fb_type in fallback_types:
                logger.info(f'[{region}/{sku}] 自动尝试替代模型: {fb_type}')
                # 删除当前模型的回测结果和模型记录
                ForecastResult.objects.filter(
                    sales_region_code=region, material_number=sku, model_used=model_type
                ).delete()
                forecast_model.delete()
                # 递归训练替代模型
                success, err = train_single_combination(region, sku, fb_type, time_range,
                                                        hierarchy_level=hierarchy_level,
                                                        category_code=category_code)
                if success:
                    logger.info(f'[{region}/{sku}] 替代模型 {fb_type} 训练成功')
                    return True, None
                else:
                    logger.warning(f'[{region}/{sku}] 替代模型 {fb_type} 失败: {err}')
            # 所有替代模型均失败，原始模型也已被删除
            err_msg = f'模型 {model_type} 及所有替代模型 {fallback_types} 均不达标'
            logger.error(f'[{region}/{sku}] {err_msg}')
            return False, err_msg
    else:
        naive_mae = mae
        mase = 1.0

    return True, None
