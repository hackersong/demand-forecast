from apps.user_management.roles import WRITE_ROLES, role_required
from django.utils.translation import gettext
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import JsonResponse
from django.contrib import messages
import json
import logging
import pandas as pd
import numpy as np
from datetime import timedelta

logger = logging.getLogger('model_training')
from collections import OrderedDict
from apps.data_management.models import SalesData
from .models import ForecastModel, ForecastResult
from .services.constants import (
    CATEGORY_FIELD_MAP,
    HIERARCHY_LABEL_MAP,
    LEVEL_FIELD_MAP,
    VALID_MODEL_TYPES,
)
# The training orchestration lives in services/trainer.py; this re-export
# keeps `from apps.model_training.views import train_single_combination`
# (apps/demand_prediction/views.py) working.
from .services.trainer import train_single_combination
from django.contrib.auth.decorators import login_required
from django.db.models import Min, Max

# ──────────────────────────────────────────────
# 产品层级映射
# ──────────────────────────────────────────────




def _build_category_cascade_tree():
    """
    从 ProductData 构建分类级联关系树，供前端动态渲染级联多选下拉框。
    返回 JSON 可序列化的 dict。
    结构说明：
      - all_l1 ~ all_l5: 各自层级全部不重复值（已排序）
      - all_skus: 所有物料号
      - l1_to_l2: {l1值: [所属l2值列表]}
      - l2_to_l3, l3_to_l4, l4_to_l5, l5_to_sku: 逐级映射
      - sku_info: {sku: {name, l1, l2, l3, l4, l5}}
    """
    from apps.data_management.models import ProductData as PD

    products = PD.objects.all().values(
        'material_number', 'material_name',
        'product_category_level1', 'product_category_level2',
        'product_category_level3', 'product_category_level4', 'product_category_level5'
    )

    l1_set, l2_set, l3_set, l4_set, l5_set = set(), set(), set(), set(), set()
    l1_to_l2, l2_to_l3, l3_to_l4, l4_to_l5, l5_to_sku = {}, {}, {}, {}, {}
    sku_info = {}

    for p in products:
        l1 = (p['product_category_level1'] or '').strip()
        l2 = (p['product_category_level2'] or '').strip()
        l3 = (p['product_category_level3'] or '').strip()
        l4 = (p['product_category_level4'] or '').strip()
        l5 = (p['product_category_level5'] or '').strip()
        sku = p['material_number']
        name = (p['material_name'] or '').strip()

        sku_info[sku] = {'name': name, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'l5': l5}

        if l1:
            l1_set.add(l1)
            l1_to_l2.setdefault(l1, set())
            if l2:
                l1_to_l2[l1].add(l2)
                l2_set.add(l2)
                l2_to_l3.setdefault(l2, set())
                if l3:
                    l2_to_l3[l2].add(l3)
                    l3_set.add(l3)
                    l3_to_l4.setdefault(l3, set())
                    if l4:
                        l3_to_l4[l3].add(l4)
                        l4_set.add(l4)
                        l4_to_l5.setdefault(l4, set())
                        if l5:
                            l4_to_l5[l4].add(l5)
                            l5_set.add(l5)
                            l5_to_sku.setdefault(l5, set())
                            l5_to_sku[l5].add(sku)
                        else:
                            l4_to_l5[l4].add('')
                    else:
                        l3_to_l4[l3].add('')
                else:
                    l2_to_l3[l2].add('')
            else:
                l1_to_l2[l1].add('')
        else:
            sku_info[sku] = {'name': name, 'l1': '', 'l2': '', 'l3': '', 'l4': '', 'l5': ''}

    # 如果存在没有父级分类的子级，补充到对应映射中
    for sku, info in sku_info.items():
        if info['l2'] and info['l2'] not in l2_to_l3:
            l2_to_l3.setdefault(info['l2'], set()).add(info['l3'])
        if info['l3'] and info['l3'] not in l3_to_l4:
            l3_to_l4.setdefault(info['l3'], set()).add(info['l4'])
        if info['l4'] and info['l4'] not in l4_to_l5:
            l4_to_l5.setdefault(info['l4'], set()).add(info['l5'])
        if info['l5'] and info['l5'] not in l5_to_sku:
            l5_to_sku.setdefault(info['l5'], set()).add(sku)

    def _sorted_list(s):
        return sorted([x for x in s if x])

    return {
        'all_l1': _sorted_list(l1_set),
        'all_l2': _sorted_list(l2_set),
        'all_l3': _sorted_list(l3_set),
        'all_l4': _sorted_list(l4_set),
        'all_l5': _sorted_list(l5_set),
        'all_skus': sorted(sku_info.keys()),
        'l1_to_l2': {k: _sorted_list(v) for k, v in l1_to_l2.items()},
        'l2_to_l3': {k: _sorted_list(v) for k, v in l2_to_l3.items()},
        'l3_to_l4': {k: _sorted_list(v) for k, v in l3_to_l4.items()},
        'l4_to_l5': {k: _sorted_list(v) for k, v in l4_to_l5.items()},
        'l5_to_sku': {k: _sorted_list(v) for k, v in l5_to_sku.items()},
        'sku_info': sku_info,
    }


def _resolve_training_items(hierarchy_level, selected_l1, selected_l2, selected_l3,
                            selected_l4, selected_l5, selected_skus):
    """
    根据层级和分类/ SKU 筛选条件，解析出最终要训练的项目代码列表。
    返回: [(item_code, category_code(可为None)), ...]
    """
    from apps.data_management.models import ProductData as PD

    if hierarchy_level == 'sku':
        qs = PD.objects.all()
        if selected_l1:
            qs = qs.filter(product_category_level1__in=selected_l1)
        if selected_l2:
            qs = qs.filter(product_category_level2__in=selected_l2)
        if selected_l3:
            qs = qs.filter(product_category_level3__in=selected_l3)
        if selected_l4:
            qs = qs.filter(product_category_level4__in=selected_l4)
        if selected_l5:
            qs = qs.filter(product_category_level5__in=selected_l5)
        if selected_skus:
            qs = qs.filter(material_number__in=selected_skus)
        sku_list = list(qs.values_list('material_number', flat=True).distinct().order_by('material_number'))
        return [(s, None) for s in sku_list]
    else:
        target_field = LEVEL_FIELD_MAP[hierarchy_level]
        qs = PD.objects.all()
        if selected_l1:
            qs = qs.filter(product_category_level1__in=selected_l1)
        if selected_l2:
            qs = qs.filter(product_category_level2__in=selected_l2)
        if selected_l3:
            qs = qs.filter(product_category_level3__in=selected_l3)
        if selected_l4:
            qs = qs.filter(product_category_level4__in=selected_l4)
        if selected_l5:
            qs = qs.filter(product_category_level5__in=selected_l5)
        cat_values = list(qs.exclude(**{target_field: ''})
                          .values_list(target_field, flat=True).distinct().order_by(target_field))
        return [(v, v) for v in cat_values]


def _get_sales_date_range():
    """获取 SalesData 表中的最早和最晚日期，返回 (min_date_str, max_date_str)"""
    agg = SalesData.objects.aggregate(min_d=Min('date'), max_d=Max('date'))
    min_d = agg.get('min_d')
    max_d = agg.get('max_d')
    if min_d and max_d:
        return min_d.strftime('%Y-%m'), max_d.strftime('%Y-%m')
    return '', ''




def _get_available_model_types():
    """检测当前环境可用的模型类型（只需将算法名称加入 VALID_MODEL_TYPES 即可启用）"""
    # 基础模型（无需额外依赖，始终可用）
    types = ['holt-winters', 'moving_avg', 'weighted_avg', 'seasonal_naive']
    # 自定义实现，无需额外依赖
    types.append('croston')
    types.append('sba')
    try:
        import sklearn.ensemble
        types.append('random_forest')
    except ImportError:
        pass
    try:
        import statsmodels
        types.append('arima')
        types.append('arimax')
        types.append('exp_smooth')
    except ImportError:
        pass
    try:
        import xgboost
        types.append('xgboost')
    except ImportError:
        pass
    try:
        import prophet
        types.append('prophet')
    except ImportError:
        pass
    try:
        import tensorflow as _tf
        types.append('lstm')
        types.append('deepar')
    except ImportError:
        pass
    return types


# ──────── 算法预测辅助函数 ────────







# ──────── Croston / SBA 季节性增强 ────────















# ──────── 季节性特征工程 ────────




# ──────── 数据清洗与预处理 ────────



# ──────── 模型融合 ────────


def _try_ensemble(region, sku, trained_model_types):
    """
    对已训练的多个模型进行加权融合 + 季节性偏差校正。

    核心改进（解决1个月时滞问题）：
    1. 逆MAE加权 —— 准确率高的模型获得更大权重
    2. 季节性偏差校正 —— 对每月计算系统性偏差(pred-actual)，校正后消除时滞
    3. 趋势-季节分解融合 —— 用实际数据的季节性替代模型预测的季节相位

    返回: (ensemble_model: ForecastModel | None, accuracy: float | None)
    """
    # ── Step 1: 收集各模型预测值，按日期和模型分组 ──
    results_by_date = {}       # {date: {'forecasts': {mt: val}, 'actual': val}}
    model_pairs = {}           # {mt: [(date, pred, actual), ...]}

    for mt in trained_model_types:
        mt_pairs = []
        for fr in ForecastResult.objects.filter(
            sales_region_code=region,
            material_number=sku,
            model_used=mt
        ).order_by('date'):
            dk = fr.date
            pred_val = float(fr.forecast_quantity)
            actual_val = float(fr.actual_quantity) if fr.actual_quantity is not None else None

            if dk not in results_by_date:
                results_by_date[dk] = {'forecasts': {}, 'actual': actual_val}
            results_by_date[dk]['forecasts'][mt] = pred_val

            if actual_val is not None:
                mt_pairs.append((dk, pred_val, actual_val))

        if len(mt_pairs) >= 3:
            model_pairs[mt] = mt_pairs

    if not results_by_date:
        return None, None

    sorted_dates = sorted(results_by_date.keys())

    # ── Step 2: 计算各模型的 MAE，用于逆MAE加权 ──
    model_mae = {}
    for mt, pairs in model_pairs.items():
        errs = [abs(p - a) for _, p, a in pairs]
        model_mae[mt] = float(np.mean(errs))

    if not model_mae:
        return None, None

    # ── Step 3: 逆MAE加权 + 平滑处理 ──
    epsilon = 1e-6
    total_inv_mae = sum(1.0 / (mae_val + epsilon) for mae_val in model_mae.values())
    model_weights = {
        mt: (1.0 / (mae + epsilon)) / total_inv_mae
        for mt, mae in model_mae.items()
    }

    # ── Step 4: 按日期计算加权平均预测 ──
    raw_ensemble = []  # [(date, weighted_pred), ...]
    for dk in sorted_dates:
        data = results_by_date[dk]
        weighted_sum = 0.0
        weight_sum = 0.0
        for mt, pred in data['forecasts'].items():
            w = model_weights.get(mt, 0.0)
            if w > 0:
                weighted_sum += w * pred
                weight_sum += w
        if weight_sum > 0:
            raw_pred = weighted_sum / weight_sum
        else:
            preds = list(data['forecasts'].values())
            raw_pred = sum(preds) / len(preds) if preds else 0.0
        raw_ensemble.append((dk, max(0.0, raw_pred)))

    # ── Step 5: 趋势-季节分解 + 季节性偏差校正 ──
    # 核心思路：模型捕捉趋势较好，但季节性相位常偏移1个月。
    # 解决方案：提取实际数据的季节模式，与模型趋势融合。
    dates_with_actual = [(dk, float(results_by_date[dk]['actual']))
                         for dk in sorted_dates
                         if results_by_date[dk]['actual'] is not None]

    if len(dates_with_actual) < 12:
        # 数据不足，跳过季节性校正，仅用加权平均
        ensemble_forecasts_list = raw_ensemble
    else:
        # 5a. 构建实际销量的月度数组（按时间排序）
        act_dates = [d[0] for d in dates_with_actual]
        act_values = np.array([d[1] for d in dates_with_actual], dtype=float)

        # 5b. 拟合实际销量的趋势线
        act_x = np.arange(len(act_values), dtype=float)
        act_trend_slope, act_trend_intercept = np.polyfit(act_x, act_values, 1)
        act_trend = act_trend_intercept + act_trend_slope * act_x

        # 5c. 从实际数据提取各月季节分量（去趋势后的残差均值）
        act_seasonal = np.zeros(12, dtype=float)
        act_seasonal_counts = np.zeros(12, dtype=int)
        for i, (dk, _) in enumerate(dates_with_actual):
            m_idx = dk.month - 1
            act_seasonal[m_idx] += (act_values[i] - act_trend[i])
            act_seasonal_counts[m_idx] += 1
        for m in range(12):
            if act_seasonal_counts[m] >= 1:
                act_seasonal[m] /= act_seasonal_counts[m]

        # 5d. 计算加权平均预测的每月系统性偏差 (pred - actual)
        monthly_bias = np.zeros(12, dtype=float)
        monthly_bias_count = np.zeros(12, dtype=int)
        for dk, raw_pred in raw_ensemble:
            act = results_by_date[dk]['actual']
            if act is not None:
                m_idx = dk.month - 1
                monthly_bias[m_idx] += (raw_pred - float(act))
                monthly_bias_count[m_idx] += 1

        avg_actual = float(np.mean(act_values))
        for m in range(12):
            if monthly_bias_count[m] >= 2:
                monthly_bias[m] /= monthly_bias_count[m]
                # 限制校正幅度 ≤ 月均销量的30%
                cap = avg_actual * 0.3
                monthly_bias[m] = np.clip(monthly_bias[m], -cap, cap)

        # 5e. 趋势-季节融合：模型趋势 + 实际季节分量（混合比例0.6）
        # 目的：保留模型对趋势的判断，但用实际季节模式纠正相位
        raw_values = np.array([p[1] for p in raw_ensemble], dtype=float)
        raw_x = np.arange(len(raw_values), dtype=float)
        raw_trend_slope, raw_trend_intercept = np.polyfit(raw_x, raw_values, 1)
        raw_trend_line = raw_trend_intercept + raw_trend_slope * raw_x

        # 也提取原始预测的季节分量
        raw_seasonal = np.zeros(12, dtype=float)
        raw_seasonal_counts = np.zeros(12, dtype=int)
        for i, (dk, raw_pred) in enumerate(raw_ensemble):
            m_idx = dk.month - 1
            raw_seasonal[m_idx] += (raw_values[i] - raw_trend_line[i])
            raw_seasonal_counts[m_idx] += 1
        for m in range(12):
            if raw_seasonal_counts[m] >= 1:
                raw_seasonal[m] /= raw_seasonal_counts[m]

        # 融合比例：后期月份用更多实际季节分量（实际数据的季节性更可靠）
        season_blend = 0.6  # 60%实际季节 + 40%模型季节

        ensemble_forecasts_list = []
        for i, (dk, raw_pred) in enumerate(raw_ensemble):
            m_idx = dk.month - 1

            # 趋势部分 + 混合季节部分 - 偏差校正
            blended_seasonal = (season_blend * act_seasonal[m_idx] +
                                (1.0 - season_blend) * raw_seasonal[m_idx])
            trend_part = raw_trend_line[i]

            # 偏差校正（进一步消除系统性偏差）
            bias_corr = monthly_bias[m_idx] if monthly_bias_count[m_idx] >= 2 else 0.0

            adjusted_pred = trend_part + blended_seasonal - bias_corr
            ensemble_forecasts_list.append((dk, max(0.0, adjusted_pred)))

    # ── Step 6: 评估准确率 ──
    paired = []
    for dk, f in ensemble_forecasts_list:
        act = results_by_date[dk]['actual']
        if act is not None:
            paired.append((f, float(act)))

    if len(paired) < 3:
        return None, None

    forecast_vals = [p[0] for p in paired]
    actual_vals = [p[1] for p in paired]
    mae = float(np.mean([abs(f - a) for f, a in zip(forecast_vals, actual_vals)]))
    rmse = float(np.sqrt(np.mean([(f - a) ** 2 for f, a in zip(forecast_vals, actual_vals)])))
    f_arr = np.array(forecast_vals)
    a_arr = np.array(actual_vals)
    pct_errs = np.where(a_arr > 0, np.abs(a_arr - f_arr) / a_arr, 0.0)
    pct_errs = np.where((a_arr == 0) & (f_arr > 0), 1.0, pct_errs)
    accuracy = max(0.0, (1.0 - np.mean(pct_errs)) * 100.0) if len(pct_errs) > 0 else 0.0

    # 获取训练日期范围（从已有模型中获取）
    existing_models = ForecastModel.objects.filter(
        sales_region_code=region,
        material_number=sku,
        model_type__in=trained_model_types
    )
    train_start = existing_models.aggregate(Min('training_start_date'))['training_start_date__min']
    train_end = existing_models.aggregate(Max('training_end_date'))['training_end_date__max']
    is_overfit = mae > np.mean(actual_vals) * 0.3 if mae > 0 else False

    model_name = f"{region}_{sku}_ensemble"
    fm, _ = ForecastModel.objects.update_or_create(
        sales_region_code=region,
        material_number=sku,
        model_type='ensemble',
        defaults={
            'model_name': model_name,
            'training_start_date': train_start,
            'training_end_date': train_end,
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'is_overfit': is_overfit
        }
    )

    # 保存 ensemble 的 ForecastResult
    ForecastResult.objects.filter(
        sales_region_code=region,
        material_number=sku,
        model_used='ensemble'
    ).delete()
    for dk, f_val in ensemble_forecasts_list:
        ForecastResult.objects.create(
            sales_region_code=region,
            material_number=sku,
            date=dk,
            forecast_quantity=f_val,
            actual_quantity=results_by_date[dk]['actual'],
            model_used='ensemble'
        )

    logger.info(
        f'[{region}/{sku}] Ensemble融合完成: '
        f'加权模型={list(model_weights.keys())}, '
        f'准确率={accuracy:.2f}%'
    )

    return fm, accuracy


# ──────── 过拟合重训参数 ────────



# ──────── 后测不达标时的替代模型优先级 ────────







@login_required
@role_required(*WRITE_ROLES)
def train_model(request):
    """模型训练视图 —— 级联多选分类筛选 + 自动尝试多种算法并选择最优模型"""
    selected_regions = []
    selected_skus = []
    selected_hierarchy = 'sku'
    selected_start_date = ''
    selected_end_date = ''
    # 级联分类筛选项
    selected_l1, selected_l2, selected_l3, selected_l4, selected_l5 = [], [], [], [], []
    training_errors = []

    # ── 获取日期范围默认值 ──
    default_start, default_end = _get_sales_date_range()

    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # ── 解析表单 ──
        selected_regions = [r.strip() for r in request.POST.getlist('region') if r.strip()]
        selected_hierarchy = request.POST.get('hierarchy_level', 'sku')
        selected_start_date = request.POST.get('start_date', default_start)
        selected_end_date = request.POST.get('end_date', default_end)
        selected_l1 = [v.strip() for v in request.POST.getlist('category_l1') if v.strip()]
        selected_l2 = [v.strip() for v in request.POST.getlist('category_l2') if v.strip()]
        selected_l3 = [v.strip() for v in request.POST.getlist('category_l3') if v.strip()]
        selected_l4 = [v.strip() for v in request.POST.getlist('category_l4') if v.strip()]
        selected_l5 = [v.strip() for v in request.POST.getlist('category_l5') if v.strip()]
        selected_skus = [v.strip() for v in request.POST.getlist('sku') if v.strip()]

        logger.info(
            '[train_model] POST regions=%s l1=%s l2=%s l3=%s l4=%s l5=%s skus=%s hierarchy=%s',
            selected_regions, selected_l1, selected_l2, selected_l3,
            selected_l4, selected_l5, selected_skus, selected_hierarchy,
        )

        # ── 解析要训练的项目列表 ──
        training_items = _resolve_training_items(
            selected_hierarchy, selected_l1, selected_l2, selected_l3,
            selected_l4, selected_l5, selected_skus
        )
        logger.info('[train_model] _resolve_training_items returned %d items', len(training_items))

        training_errors = []
        if not selected_regions:
            training_errors.append('请选择至少一个销售区域')
        elif not training_items:
            logger.info(
                '[train_model] NO items found. l1=%s l2=%s l3=%s l4=%s l5=%s skus=%s hierarchy=%s',
                selected_l1, selected_l2, selected_l3, selected_l4,
                selected_l5, selected_skus, selected_hierarchy,
            )
            training_errors.append('当前筛选条件下没有可训练的项目，请调整筛选条件')
        else:
            if not selected_start_date:
                selected_start_date = default_start
            if not selected_end_date:
                selected_end_date = default_end

            available_model_types = _get_available_model_types()
            first_model_id = None

            try:
                for region in selected_regions:
                    for item_code, cat_code in training_items:
                        trained_models = []

                        for mt in available_model_types:
                            success, error_msg = train_single_combination(
                                region, item_code, mt, time_range=12,
                                hierarchy_level=selected_hierarchy,
                                category_code=cat_code,
                                start_date_str=selected_start_date,
                                end_date_str=selected_end_date,
                            )
                            if success:
                                # 注意：train_single_combination 可能通过回退链创建了
                                # 不同 model_type 的模型，因此不按 mt 过滤，取最新创建的
                                fm = ForecastModel.objects.filter(
                                    sales_region_code=region,
                                    material_number=item_code,
                                    hierarchy_level=selected_hierarchy,
                                    category_code=cat_code,
                                ).order_by('-created_at').first()
                                if fm:
                                    trained_models.append(fm)

                        if not trained_models:
                            item_label = item_code
                            training_errors.append(f'区域 {region} {item_label} 所有模型训练均失败')
                            continue

                        # 过滤掉过拟合模型，优先选择非过拟合的最优模型
                        non_overfit = [m for m in trained_models if not m.is_overfit]
                        if non_overfit:
                            best_model = max(non_overfit, key=lambda m: m.accuracy)
                        else:
                            best_model = max(trained_models, key=lambda m: m.accuracy)

                        if len(trained_models) >= 2:
                            ensemble_model, ensemble_acc = _try_ensemble(
                                region, item_code, [m.model_type for m in trained_models]
                            )
                            if ensemble_model and ensemble_acc is not None and ensemble_acc > best_model.accuracy:
                                best_model = ensemble_model

                        ForecastResult.objects.filter(
                            sales_region_code=region,
                            material_number=item_code,
                            hierarchy_level=selected_hierarchy,
                            category_code=cat_code,
                        ).exclude(model_used=best_model.model_type).delete()
                        ForecastModel.objects.filter(
                            sales_region_code=region,
                            material_number=item_code,
                            hierarchy_level=selected_hierarchy,
                            category_code=cat_code,
                        ).exclude(id=best_model.id).delete()

                        if first_model_id is None:
                            first_model_id = best_model.pk
            except Exception as e:
                logger.exception('模型训练过程中发生异常')
                training_errors.append(f'训练过程出错: {str(e)}')
                if is_ajax:
                    return JsonResponse({'success': False, 'errors': training_errors})

            # ── 多区域多SKU 自动加命名前缀 ──
            is_multi_region = len(selected_regions) > 1
            is_multi_sku = len(training_items) > 1
            if is_multi_region or is_multi_sku:
                prefix_parts = []
                if is_multi_region:
                    prefix_parts.append('多区域')
                if is_multi_sku:
                    prefix_parts.append('多sku')
                name_prefix = f"{'多区域多sku' if (is_multi_region and is_multi_sku) else ''.join(prefix_parts)}模型_"
                renamed = 0
                for fm_obj in ForecastModel.objects.filter(
                    sales_region_code__in=selected_regions,
                    material_number__in=[t[0] for t in training_items],
                    hierarchy_level=selected_hierarchy,
                ):
                    if not fm_obj.model_name.startswith(name_prefix):
                        fm_obj.model_name = name_prefix + fm_obj.model_name
                        fm_obj.save(update_fields=['model_name'])
                        renamed += 1
                if renamed:
                    logger.info('[train_model] Renamed %d models with prefix "%s"', renamed, name_prefix)

            if first_model_id:
                redirect_url = f"{reverse('model:results')}?model_id={first_model_id}"
                if is_ajax:
                    return JsonResponse({'success': True, 'redirect_url': redirect_url, 'message': gettext('模型训练完成')})
                return redirect(redirect_url)

            if not training_errors:
                training_errors.append('所有选定区域和分类的模型训练均未成功，请检查数据是否充足')

            if is_ajax:
                return JsonResponse({'success': False, 'errors': training_errors})

        # 早期校验错误的 AJAX 响应（未进入 else 分支的情况，如无区域/无训练项目）
        if is_ajax and training_errors:
            return JsonResponse({
                'success': False,
                'errors': training_errors,
                # 附带诊断信息方便排查
                'debug': {
                    'regions': selected_regions,
                    'training_items_count': len(training_items),
                    'l1': selected_l1, 'l2': selected_l2, 'l3': selected_l3,
                    'l4': selected_l4, 'l5': selected_l5, 'skus': selected_skus,
                }
            })

    # ── GET: 准备页面数据 ──
    raw_regions = SalesData.objects.values_list('sales_region_code', flat=True).distinct()
    regions = sorted(set(r.strip() for r in raw_regions if r and r.strip()))
    # ── SKU→区域 映射（用于级联：选择分类/SKU后过滤可选区域）──
    sku_region_pairs = SalesData.objects.values('product_data__material_number', 'sales_region_code').distinct()
    _sr_map = {}
    for pair in sku_region_pairs:
        sku = pair['product_data__material_number']
        region = (pair['sales_region_code'] or '').strip()
        if sku and region:
            _sr_map.setdefault(sku, set()).add(region)
    sku_regions_map = {k: sorted(v) for k, v in _sr_map.items()}
    category_tree = _build_category_cascade_tree()
    return render(request, 'model_training/train.html', {
        'regions_list': list(regions),
        'category_tree': category_tree,
        'sku_regions_map': sku_regions_map,
        'default_start_date': default_start,
        'default_end_date': default_end,
        'selected_regions': selected_regions,
        'selected_skus': selected_skus,
        'selected_start_date': selected_start_date or default_start,
        'selected_end_date': selected_end_date or default_end,
        'selected_l1': selected_l1, 'selected_l2': selected_l2,
        'selected_l3': selected_l3, 'selected_l4': selected_l4, 'selected_l5': selected_l5,
        'active_tab': 'model',
        'training_errors': training_errors
    })


@login_required
def model_results(request):
    """模型结果视图"""
    model_id = request.GET.get('model_id', '')
    if not model_id:
        messages.error(request, gettext('未指定模型'))
        return redirect('model:train')
    
    try:
        forecast_model = ForecastModel.objects.get(id=model_id)
    except ForecastModel.DoesNotExist:
        messages.error(request, gettext('模型不存在'))
        return redirect('model:train')
    
    results = ForecastResult.objects.filter(
        sales_region_code=forecast_model.sales_region_code,
        material_number=forecast_model.material_number,
        hierarchy_level=forecast_model.hierarchy_level,
        category_code=forecast_model.category_code,
        model_used=forecast_model.model_type
    ).order_by('date')

    hl = forecast_model.hierarchy_level or 'sku'
    if hl == 'sku':
        sales_history = SalesData.objects.filter(
            sales_region_code=forecast_model.sales_region_code,
            product_data__material_number=forecast_model.material_number
        ).order_by('date')
    else:
        field_name = CATEGORY_FIELD_MAP.get(hl, 'product_data__product_category_level1')
        sales_history = SalesData.objects.filter(
            sales_region_code=forecast_model.sales_region_code,
            **{field_name: forecast_model.material_number}
        ).order_by('date')

    actual_by_month = {}
    for s in sales_history:
        month = s.date.strftime('%Y-%m')
        actual_by_month[month] = actual_by_month.get(month, 0.0) + float(s.sales_quantity)

    result_by_month = OrderedDict()
    for r in results:
        month = r.date.strftime('%Y-%m')
        result_by_month[month] = {
            'forecast': float(r.forecast_quantity),
            'actual': float(r.actual_quantity) if r.actual_quantity is not None else None
        }

    if sales_history.exists():
        current_month = sales_history.first().date.replace(day=1)
    else:
        current_month = forecast_model.training_start_date.replace(day=1)
    if results.exists():
        last_month = results.last().date.replace(day=1)
    else:
        last_month = forecast_model.training_end_date.replace(day=1)

    chart_labels = []
    while current_month <= last_month:
        chart_labels.append(current_month.strftime('%Y-%m'))
        year = current_month.year + (1 if current_month.month == 12 else 0)
        month = 1 if current_month.month == 12 else current_month.month + 1
        current_month = current_month.replace(year=year, month=month)

    # ── 从 ForecastResult 获取产品/分类展示信息 ──
    first_result = results.first()
    display_material_name = ''
    display_category_l1 = ''
    display_category_l2 = ''
    display_category_l3 = ''
    display_category_l4 = ''
    display_category_l5 = ''
    if first_result:
        display_material_name = first_result.material_name or ''
        display_category_l1 = first_result.category_l1 or ''
        display_category_l2 = first_result.category_l2 or ''
        display_category_l3 = first_result.category_l3 or ''
        display_category_l4 = first_result.category_l4 or ''
        display_category_l5 = first_result.category_l5 or ''

    result_list = []
    for month in chart_labels:
        forecast_value = result_by_month.get(month, {}).get('forecast')
        actual_value = result_by_month.get(month, {}).get('actual')
        if actual_value is None:
            actual_value = actual_by_month.get(month)

        diff_value = None
        if forecast_value is not None and actual_value is not None:
            diff_value = round(forecast_value - actual_value, 2)

        result_list.append({
            'date': month,
            'sku': forecast_model.material_number,
            'material_name': display_material_name,
            'category_l1': display_category_l1,
            'category_l2': display_category_l2,
            'category_l3': display_category_l3,
            'category_l4': display_category_l4,
            'category_l5': display_category_l5,
            'forecast': round(forecast_value, 2) if forecast_value is not None else None,
            'actual': round(actual_value, 2) if actual_value is not None else None,
            'diff': diff_value
        })

    actual_items = [item for item in result_list if item['forecast'] is not None and item['actual'] is not None]
    actual_count = len(actual_items)
    average_actual = np.mean([item['actual'] for item in actual_items]) if actual_items else None
    bias = round(np.mean([item['forecast'] - item['actual'] for item in actual_items]), 2) if actual_items else None
    mape = round(np.mean([abs(item['forecast'] - item['actual']) / item['actual'] * 100 for item in actual_items if item['actual'] != 0]), 2) if actual_items else None
    level = None
    if mape is not None:
        if mape < 10:
            level = '优秀'
        elif mape < 20:
            level = '良好'
        elif mape < 30:
            level = '一般'
        else:
            level = '差'
    error_stats = {
        'mae': round(np.mean([abs(item['forecast'] - item['actual']) for item in actual_items]), 2) if actual_items else None,
        'rmse': round(np.sqrt(np.mean([(item['forecast'] - item['actual']) ** 2 for item in actual_items])), 2) if actual_items else None,
        'max_error': round(max([abs(item['forecast'] - item['actual']) for item in actual_items]), 2) if actual_items else None,
        'bias': bias,
        'bias_pct': round(bias / average_actual * 100, 2) if bias is not None and average_actual else None,
        'mape': mape,
        'level': level,
        'actual_count': actual_count
    }

    # 基于全部历史回测数据更新 ForecastModel 的 MAE/RMSE/准确率
    if actual_items:
        all_for = np.array([item['forecast'] for item in actual_items])
        all_act = np.array([item['actual'] for item in actual_items])
        # 用 actual 做分母（与 _eval_metrics 保持一致），避免 forecast 极小时误差爆炸
        pct_errs = np.where(all_act > 0, np.abs(all_act - all_for) / all_act, 0.0)
        pct_errs = np.where((all_act == 0) & (all_for > 0), 1.0, pct_errs)
        recalc_accuracy = round(max(0.0, (1.0 - np.mean(pct_errs)) * 100.0), 2) if len(pct_errs) > 0 else 0.0
        forecast_model.accuracy = recalc_accuracy
        forecast_model.mae = error_stats['mae']
        forecast_model.rmse = error_stats['rmse']
        forecast_model.save(update_fields=['accuracy', 'mae', 'rmse'])

    sales_list = []
    for s in sales_history:
        sales_list.append({
            'date': s.date.strftime('%Y-%m'),
            'quantity': round(s.sales_quantity, 2)
        })

    chart_forecasts_backtest = [r['forecast'] if r['forecast'] is not None and r['actual'] is not None else None for r in result_list]
    chart_forecasts_future = [r['forecast'] if r['forecast'] is not None and r['actual'] is None else None for r in result_list]
    chart_actuals = [r['actual'] for r in result_list]

    # 计算 MAE/RMSE 相对月均销量的占比，用于参考范围展示
    mae_ratio_pct = None
    rmse_ratio_pct = None
    if average_actual and average_actual > 0:
        if error_stats['mae'] is not None:
            mae_ratio_pct = round(error_stats['mae'] / average_actual * 100, 1)
        if error_stats['rmse'] is not None:
            rmse_ratio_pct = round(error_stats['rmse'] / average_actual * 100, 1)

    # 朴素基线MAE："用上月值预测当月"，MASE < 1.0 说明模型优于朴素基线
    naive_mae = None
    mase = None
    if actual_items and len(actual_items) > 1:
        naive_errs = [abs(actual_items[i]['actual'] - actual_items[i-1]['actual'])
                      for i in range(1, len(actual_items))]
        naive_mae = round(float(np.mean(naive_errs)), 2)
        if error_stats['mae'] and naive_mae > 0:
            mase = round(error_stats['mae'] / naive_mae, 3)

    context = {
        'model': forecast_model,
        'results': result_list,
        'sales_data': sales_list,
        'chart_labels': json.dumps(chart_labels, ensure_ascii=False),
        'chart_forecasts_backtest': json.dumps(chart_forecasts_backtest),
        'chart_forecasts_future': json.dumps(chart_forecasts_future),
        'chart_actuals': json.dumps(chart_actuals),
        'error_stats': error_stats,
        'avg_monthly_sales': round(average_actual, 2) if average_actual else None,
        'mae_ratio_pct': mae_ratio_pct,
        'rmse_ratio_pct': rmse_ratio_pct,
        'naive_mae': naive_mae,
        'mase': mase,
        'active_tab': 'model'
    }

    return render(request, 'model_training/model_results.html', context)
