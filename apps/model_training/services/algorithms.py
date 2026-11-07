"""Forecasting algorithms (pure computation, no Django).

Moved verbatim from views.py. The two `_seasonal_decompose` definitions
keep their original order on purpose: the second definition shadows the
first at module level, which is exactly the behaviour the training path
relied on before the extraction.
"""
import numpy as np
import pandas as pd

def _arimax_predict(series, periods=12):
    """ARIMAX：ARIMA + 时间趋势作为外生变量"""
    from statsmodels.tsa.arima.model import ARIMA
    n = len(series)
    exog = np.arange(n, dtype=float).reshape(-1, 1)
    model = ARIMA(series, exog=exog, order=(1, 1, 1)).fit()
    future_exog = np.arange(n, n + periods, dtype=float).reshape(-1, 1)
    return np.maximum(0.0, model.forecast(steps=periods, exog=future_exog))

def _prophet_predict(series, periods=12):
    """Prophet 时序预测"""
    from prophet import Prophet
    df = pd.DataFrame({'ds': series.index.to_timestamp() if hasattr(series.index, 'to_timestamp') else pd.to_datetime(series.index.values),
                       'y': series.values.astype(float)})
    model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=len(series) >= 18)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    fc = model.predict(future)
    return np.maximum(0.0, fc.tail(periods)['yhat'].values.astype(float))

def _random_forest_predict(monthly_data, periods=12):
    """随机森林预测"""
    from sklearn.ensemble import RandomForestRegressor
    monthly_df = monthly_data.reset_index()
    monthly_df['month_index'] = np.arange(len(monthly_df), dtype=float)
    monthly_df['month_of_year'] = monthly_df['date'].dt.month.astype(float)
    X = monthly_df[['month_index', 'month_of_year']].values
    y = monthly_df['sales_quantity'].astype(float).values
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)
    last_idx = len(monthly_df)
    future_idx = np.arange(last_idx, last_idx + periods, dtype=float)
    future_months = np.array([(monthly_df['date'].iloc[-1].month + i) % 12 + 1
                              for i in range(1, periods + 1)], dtype=float)
    future_X = np.column_stack([future_idx, future_months])
    return np.maximum(0.0, model.predict(future_X))

def _seasonal_decompose(values, season_length):
    """
    季节分解：基于中位数比率法计算季节指数，并返回去季节化序列。

    使用中位数（而非均值）以抵抗异常值的干扰。
    返回: (seasonal_indices: np.ndarray, deseasonalized: np.ndarray)
    """
    n = len(values)
    seasonal_indices = np.ones(season_length)

    for s in range(season_length):
        season_vals = values[s::season_length]
        if len(season_vals) > 0:
            nonzero_season = season_vals[season_vals > 0]
            nonzero_all = values[values > 0]
            if len(nonzero_season) > 0 and len(nonzero_all) > 0:
                s_median = float(np.median(nonzero_season))
                o_median = float(np.median(nonzero_all))
                if o_median > 0:
                    seasonal_indices[s] = s_median / o_median

    # 归一化使季节指数均值为 1
    si_mean = float(np.mean(seasonal_indices))
    if si_mean > 0:
        seasonal_indices = seasonal_indices / si_mean

    # 去季节化
    deseasonalized = values.copy()
    for i in range(n):
        s = i % season_length
        if seasonal_indices[s] > 0.01:
            deseasonalized[i] = values[i] / seasonal_indices[s]

    return seasonal_indices, deseasonalized

def _croston_core(values, alpha, apply_sba=False):
    """
    Croston/SBA 核心算法：对（可能已去季节化）的序列计算需求率。
    apply_sba=True 时应用 SBA 偏差修正。
    """
    nonzero_idx = np.where(values > 0)[0]
    if len(nonzero_idx) < 2:
        return float(np.mean(values))

    demands = values[nonzero_idx]
    intervals = np.diff(nonzero_idx).astype(float)
    # 最后一段间隔用最近 3 段均值估计（或整体均值兜底）
    last_interval_est = float(np.mean(intervals[-3:])) if len(intervals) >= 3 else float(np.mean(intervals))
    intervals = np.append(intervals, last_interval_est)

    sd = float(demands[0])
    si = float(intervals[0])
    for j in range(1, len(demands)):
        sd = alpha * float(demands[j]) + (1 - alpha) * sd
        si = alpha * float(intervals[j]) + (1 - alpha) * si

    demand_rate = sd / max(si, 1.0)
    if apply_sba:
        demand_rate = (1 - alpha / 2) * demand_rate
    return max(0.0, demand_rate)

def _croston_predict(series, periods=12, alpha=0.15):
    """
    Croston 方法 - 适用于间歇性需求的时间序列。
    当数据 >= 24 个月时，自动进行季节性分解后再应用 Croston，
    解决标准 Croston 在季节性数据上预测滞后一期的固有问题。
    """
    values = np.asarray(series.values if hasattr(series, 'values') else series, dtype=float)
    n = len(values)

    # 自动季节性检测：>= 24 个月时尝试 12 个月周期分解
    if n >= 24:
        try:
            seasonal_indices, deseasonalized = _seasonal_decompose(values, season_length=12)
            base_rate = _croston_core(deseasonalized, alpha, apply_sba=False)
            forecast = np.array([
                max(0.0, base_rate * seasonal_indices[(n + p) % 12])
                for p in range(periods)
            ])
            return forecast
        except Exception:
            pass  # 回退到标准 Croston

    # 标准 Croston（数据不足 24 个月或季节性分解失败时）
    base_rate = _croston_core(values, alpha, apply_sba=False)
    return np.full(periods, base_rate)

def _sba_predict(series, periods=12, alpha=0.15):
    """
    SBA (Syntetos-Boylan Approximation) - Croston 的偏差修正版。
    同样支持自动季节性检测。
    """
    values = np.asarray(series.values if hasattr(series, 'values') else series, dtype=float)
    n = len(values)

    if n >= 24:
        try:
            seasonal_indices, deseasonalized = _seasonal_decompose(values, season_length=12)
            base_rate = _croston_core(deseasonalized, alpha, apply_sba=True)
            forecast = np.array([
                max(0.0, base_rate * seasonal_indices[(n + p) % 12])
                for p in range(periods)
            ])
            return forecast
        except Exception:
            pass

    base_rate = _croston_core(values, alpha, apply_sba=True)
    return np.full(periods, base_rate)

def _lstm_predict(series, periods=12, lookback=6, epochs=60):
    """LSTM 神经网络预测"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    values = np.asarray(series.values if hasattr(series, 'values') else series, dtype=float)
    mean_v, std_v = np.mean(values), np.std(values) + 1e-8
    scaled = (values - mean_v) / std_v
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(32, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
        LSTM(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=min(4, len(X)))
    last_seq = scaled[-lookback:].copy()
    forecasts = []
    for _ in range(periods):
        inp = last_seq.reshape(1, lookback, 1)
        pred = float(model.predict(inp, verbose=0)[0, 0])
        forecasts.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    return np.maximum(0.0, np.array(forecasts) * std_v + mean_v)

def _deepar_predict(series, periods=12, lookback=6, epochs=100):
    """DeepAR 简化版 - 更深层的 LSTM 架构，适用于概率预测"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    values = np.asarray(series.values if hasattr(series, 'values') else series, dtype=float)
    mean_v, std_v = np.mean(values), np.std(values) + 1e-8
    scaled = (values - mean_v) / std_v
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        LSTM(32, activation='relu', return_sequences=True),
        Dropout(0.1),
        LSTM(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.005), loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0, batch_size=min(4, len(X)))
    last_seq = scaled[-lookback:].copy()
    forecasts = []
    for _ in range(periods):
        inp = last_seq.reshape(1, lookback, 1)
        pred = float(model.predict(inp, verbose=0)[0, 0])
        forecasts.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    return np.maximum(0.0, np.array(forecasts) * std_v + mean_v)

def _seasonal_decompose(series):
    """
    对月度时序进行趋势-季节分解，提取趋势线和各月季节指数。

    参数:
        series: pd.Series，index 为日期，values 为月度销量

    返回:
        dict: {
            'trend': np.ndarray,         # 趋势线 (与 series 同长度)
            'seasonal_index': np.ndarray, # 12个月的季节指数（比率，1.0=平均水平）
            'deseasonalized': np.ndarray, # 去季节化后的序列（保留趋势+残差）
            'seasonal_strength': float,   # 季节性强度 (0~1)，越高季节效应越显著
        }
    """
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    result = {
        'trend': np.zeros(n),
        'seasonal_index': np.ones(12),
        'deseasonalized': np.zeros(n),
        'seasonal_strength': 0.0,
    }

    if n < 12:
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        result['trend'] = intercept + slope * x
        result['deseasonalized'] = values.copy()
        return result

    # Step 1: 拟合线性趋势
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    trend = intercept + slope * x
    result['trend'] = trend

    # Step 2: 计算去趋势残差
    detrended = values - trend

    # Step 3: 计算各月季节分量（平均去趋势残差）
    monthly_residuals = np.zeros(12)
    monthly_counts = np.zeros(12, dtype=int)
    for i in range(n):
        if hasattr(series.index[i], 'month'):
            m = series.index[i].month - 1
        else:
            m = i % 12
        monthly_residuals[m] += detrended[i]
        monthly_counts[m] += 1

    for m in range(12):
        if monthly_counts[m] > 0:
            monthly_residuals[m] /= monthly_counts[m]

    # Step 4: 构建季节指数（比率形式，1.0 = 平均水平）
    avg_trend = np.mean(trend) if np.mean(trend) > 0 else 1.0
    seasonal_index = np.ones(12)
    for m in range(12):
        if avg_trend > 0:
            seasonal_index[m] = max(0.1, (avg_trend + monthly_residuals[m]) / avg_trend)

    result['seasonal_index'] = seasonal_index

    # Step 5: 构建去季节化序列（trend + residual only）
    deseasonalized = np.zeros(n)
    for i in range(n):
        if hasattr(series.index[i], 'month'):
            m = series.index[i].month - 1
        else:
            m = i % 12
        si = seasonal_index[m]
        if si > 0:
            deseasonalized[i] = values[i] / si
        else:
            deseasonalized[i] = values[i]
    result['deseasonalized'] = deseasonalized

    # Step 6: 季节性强度：Var(seasonal) / (Var(seasonal) + Var(residual))
    seasonal_comp = np.zeros(n)
    for i in range(n):
        if hasattr(series.index[i], 'month'):
            m = series.index[i].month - 1
        else:
            m = i % 12
        seasonal_comp[i] = monthly_residuals[m]
    residual_comp = detrended - seasonal_comp
    var_s = np.var(seasonal_comp) if len(seasonal_comp) > 0 else 0.0
    var_r = np.var(residual_comp) if len(residual_comp) > 0 else 0.0
    if var_s + var_r > 0:
        result['seasonal_strength'] = max(0.0, min(1.0, var_s / (var_s + var_r)))

    return result

def _get_retry_params(model_type, attempt):
    """
    返回第 attempt 次训练尝试的参数配置。
    attempt 0: 默认参数
    attempt 1: 简化/正则化参数（减少过拟合风险）
    attempt 2: 最简化参数（最保守配置）
    """
    params = {}

    if model_type == 'arima':
        params['arima_order'] = [(1, 1, 1), (0, 1, 1), (2, 1, 0)][attempt]

    elif model_type == 'arimax':
        params['arimax_order'] = [(1, 1, 1), (0, 1, 1), (2, 1, 0)][attempt]

    elif model_type == 'xgboost':
        configs = [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.3},
            {'n_estimators': 50,  'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 30,  'max_depth': 2, 'learning_rate': 0.05},
        ]
        params.update(configs[attempt])

    elif model_type == 'random_forest':
        configs = [
            {'n_estimators': 150, 'max_depth': None},
            {'n_estimators': 80,  'max_depth': 6},
            {'n_estimators': 40,  'max_depth': 3},
        ]
        params.update(configs[attempt])

    elif model_type == 'exp_smooth':
        # 过拟合时逐次减小季节周期，但始终保留季节性（避免退化为直线）
        configs = [
            {'seasonal_enabled': True,  'seasonal_periods': 12},
            {'seasonal_enabled': True,  'seasonal_periods': 6},
            {'seasonal_enabled': True,  'seasonal_periods': 4},
        ]
        params.update(configs[attempt])

    elif model_type == 'holt-winters':
        # Holt-Winters 过拟合时逐次减小季节周期，但始终保留季节性（避免退化为直线）
        configs = [
            {'seasonal_periods': 12, 'use_seasonal': True},
            {'seasonal_periods': 6,  'use_seasonal': True},
            {'seasonal_periods': 4,  'use_seasonal': True},
        ]
        params.update(configs[attempt])

    elif model_type == 'prophet':
        configs = [
            {'yearly_seasonality': True,  'changepoint_prior_scale': 0.05},
            {'yearly_seasonality': False, 'changepoint_prior_scale': 0.01},
            {'yearly_seasonality': False, 'changepoint_prior_scale': 0.001},
        ]
        params.update(configs[attempt])

    elif model_type == 'lstm':
        params['epochs'] = [60, 30, 20][attempt]
        params['lookback'] = [6, 4, 3][attempt]

    elif model_type == 'deepar':
        params['epochs'] = [50, 30, 20][attempt]
        params['lookback'] = [6, 4, 3][attempt]

    return params

def _get_fallback_model_types(model_type):
    """
    当模型后测准确率 < 50% 或过拟合时，按优先级返回应尝试的替代模型类型。
    优先级原则：从简单稳定模型到复杂模型，避免用一个差的模型替换另一个差的模型。
    所有链条最终收敛到 seasonal_naive / 线性回归兜底，无循环。

    返回: list[str] — 替代模型类型列表（按优先级从高到低）
    """
    # Croston/SBA 是间歇性需求专用模型，数据不匹配时回退
    if model_type in ('croston', 'sba'):
        return ['moving_avg', 'weighted_avg', 'seasonal_naive']

    # 复杂 ML 模型过拟合时回退到统计模型
    if model_type in ('xgboost', 'random_forest', 'lstm', 'deepar', 'prophet'):
        return ['holt-winters', 'moving_avg', 'seasonal_naive']

    # ARIMA/ARIMAX 回退
    if model_type in ('arima', 'arimax'):
        return ['holt-winters', 'moving_avg', 'seasonal_naive']

    # exp_smooth 是 holt-winters 的别名，同样的回退链
    if model_type == 'exp_smooth':
        return ['moving_avg', 'seasonal_naive']

    # Holt-Winters 指数平滑回退（比 ARIMA/ML 轻量，但有季节性假设）
    if model_type == 'holt-winters':
        return ['moving_avg', 'seasonal_naive']

    # moving_avg / weighted_avg → 季节性朴素（最终兜底）
    if model_type in ('moving_avg', 'weighted_avg'):
        return ['seasonal_naive']

    # 季节性朴素是最简基线，无进一步回退
    if model_type == 'seasonal_naive':
        return []

    return ['moving_avg', 'seasonal_naive']

def _compute_trend_forecast(monthly_arr, horizon=12):
    """基于线性趋势生成非恒值预测，避免简单模型产出直线"""
    arr = np.asarray(monthly_arr, dtype=float)
    if len(arr) < 2:
        val = float(arr[-1]) if len(arr) > 0 else 0.0
        return [max(0.0, val) for _ in range(horizon)]
    # 用最多最近12个月拟合线性趋势
    trend_data = arr[-min(12, len(arr)):]
    t = np.arange(len(trend_data), dtype=float)
    slope, intercept = np.polyfit(t, trend_data, 1)
    # 基准值：趋势线最后一点
    base_val = intercept + slope * (len(trend_data) - 1)
    # 外推未来 horizon 个月
    return [max(0.0, base_val + slope * (i + 1)) for i in range(horizon)]
