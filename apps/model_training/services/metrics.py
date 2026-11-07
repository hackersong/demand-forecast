"""Forecast quality metrics (pure computation, no Django)."""
import numpy as np


def _eval_metrics(actual, forecast):
    """统一的评估指标计算，返回 (mae, rmse, accuracy)
    
    accuracy 基于 WMAPE 变体：用 actual 做分母（而非 forecast），
    避免因预测值接近 0 导致百分比误差爆炸的问题。
    """
    act = np.asarray(actual, dtype=float)
    fc = np.maximum(0.0, np.asarray(forecast, dtype=float))
    mae = float(np.mean(np.abs(act - fc)))
    rmse = float(np.sqrt(np.mean((act - fc) ** 2)))
    # 用 actual 做分母（标准 WMAPE 做法），避免 forecast 极小时误差爆炸
    pct = np.where(act > 0, np.abs(act - fc) / act, 0.0)
    # actual=0 但 forecast>0：视为 100% 误差
    pct = np.where((act == 0) & (fc > 0), 1.0, pct)
    accuracy = max(0.0, (1.0 - np.mean(pct)) * 100.0) if len(pct) > 0 else 0.0
    return mae, rmse, accuracy
