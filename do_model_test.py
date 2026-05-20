import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from ml_basic_functions import test_model, get_data, ml_load_model


DATA_DICT_NAME = 'training_data'
MODEL_DICT_NAME = 'module_save'
DETAIL_RESULT_XLSX = 'test_result.xlsx'
SUMMARY_RESULT_XLSX = 'do_model_test_summary.xlsx'

TRAINING_NUM = 1000
TEST_NUM = 100
DATA_IS_RANDOM = True
VAL_RATIO = 0.1
RANDOM_SEED = 41


TEST_CASES = [
    {
        'case_name': 'old_agg_model_on_new_agg_data',
        'label': '旧 aggregate 模型 + 新 aggregate 数据',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro_decreased_base_random_1000_50000_128_0.0002',
        'data_name': 'all_training_data_case1888rte_pro_newinput_decreased_base_case1888rte_pro',
    },
    {
        'case_name': 'old_raw_model_on_new_raw_data',
        'label': '旧未 aggregate 模型 + 新未 aggregate 数据',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro_random_1000_50000_128_0.0002',
        'data_name': 'all_training_data_case1888rte_pro_newinput',
    },
    {
        'case_name': 'new_agg_model_on_new_agg_data',
        'label': '新 aggregate 模型 + 新 aggregate 数据',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro_newinput_decreased_base_random_1000_50000_128_0.0002',
        'data_name': 'all_training_data_case1888rte_pro_newinput_decreased_base',
    },
    {
        'case_name': 'new_raw_model_on_new_raw_data',
        'label': '新未 aggregate 模型 + 新未 aggregate 数据',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro_newinput_random_1000_50000_128_0.0002',
        'data_name': 'all_training_data_case1888rte_pro_newinput',
    },
]


def _format_pct(x):
    if pd.isna(x):
        return 'N/A'
    return f'{x:.3f}%'


def _calc_switch_subset_metrics(model, device, test_inputs, test_targets, recognize_constr):
    if recognize_constr:
        return {
            'test_switch_right_rate': np.nan,
            'test_switch_count': 0,
            'test_switch_ratio': np.nan,
        }

    model.eval()
    with torch.no_grad():
        test_output = model(test_inputs.to(device))

    pred = (test_output.detach().cpu().numpy() >= 0.5)
    target = (test_targets.detach().cpu().numpy() >= 0.5)

    if target.ndim != 3:
        return {
            'test_switch_right_rate': np.nan,
            'test_switch_count': 0,
            'test_switch_ratio': np.nan,
        }

    switch_mask = target[:, 1:, :] != target[:, :-1, :]
    switch_count = int(np.sum(switch_mask))
    switch_total = int(switch_mask.size)
    if switch_count == 0:
        switch_right_rate = np.nan
    else:
        switch_right_rate = float(np.mean(pred[:, 1:, :][switch_mask] == target[:, 1:, :][switch_mask]) * 100)

    return {
        'test_switch_right_rate': switch_right_rate,
        'test_switch_count': switch_count,
        'test_switch_ratio': float(switch_count * 100 / switch_total) if switch_total > 0 else np.nan,
    }


def _load_test_tensors(model_type, data_path, training_num, test_num,
                       data_is_random, cutted_by_time, recognize_constr):
    return get_data(
        model_type,
        data_path,
        test_num=test_num,
        training_num=training_num,
        data_is_random=data_is_random,
        cutted_by_time=cutted_by_time,
        recognize_constr=recognize_constr,
        random_seed=RANDOM_SEED,
        val_ratio=VAL_RATIO,
    )


def _run_single_case(case_cfg):
    model_name = case_cfg['model_name']
    data_name = case_cfg['data_name']
    data_path = os.path.join(DATA_DICT_NAME, data_name + '.npz')

    print('=' * 100)
    print(f"测试场景: {case_cfg['label']}")
    print(f"model : {model_name}")
    print(f"data  : {data_name}.npz")

    model, model_type, training_num, test_num, data_is_random, cutted_by_time, _, data_type = \
        ml_load_model(model_name, MODEL_DICT_NAME)

    recognize_constr = (data_type == 'constraints')
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    (train_inputs, train_targets,
     val_inputs, val_targets,
     test_inputs, test_targets,
     time_num, input_size, output_size,
     average_load, train_size) = _load_test_tensors(
        model_type=model_type,
        data_path=data_path,
        training_num=training_num,
        test_num=test_num,
        data_is_random=data_is_random,
        cutted_by_time=cutted_by_time,
        recognize_constr=recognize_constr,
    )

    print(f"input_size={input_size}, output_size={output_size}, time_num={time_num}, average_load={average_load:.4f}")
    print(f"train/val/test = {len(train_inputs)}/{len(val_inputs)}/{len(test_inputs)}")

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    _, test_result = test_model(
        model=model,
        criterion=criterion,
        device=device,
        loss_record=[],
        val_loss_record=[],
        inputs=train_inputs,
        targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        stop_gap=50,
        reliability_low_gate=0.1,
        reliability_high_gate=0.9,
        on_training=False,
        recognize_constr=recognize_constr,
        save_name=DETAIL_RESULT_XLSX,
        model_name=model_name,
    )
    switch_metrics = _calc_switch_subset_metrics(
        model=model,
        device=device,
        test_inputs=test_inputs,
        test_targets=test_targets,
        recognize_constr=recognize_constr,
    )

    summary = {
        'case_name': case_cfg['case_name'],
        'label': case_cfg['label'],
        'model_name': model_name,
        'data_name': data_name,
        'test_all_right_rate': test_result['all_right_rate'],
        'test_avg_right_rate': test_result['avg_right_rate'],
        'test_reliable_rate': test_result['reliable_rate'],
        'test_reliable_right_rate': test_result['reliable_right_rate'],
        'test_loss': test_result['loss'],
        **switch_metrics,
    }

    print('-' * 100)
    print(f"测试集全对率     : {_format_pct(summary['test_all_right_rate'])}")
    print(f"测试集平均正确率 : {_format_pct(summary['test_avg_right_rate'])}")
    print(f"测试集切换准确率 : {_format_pct(summary['test_switch_right_rate'])} "
          f"(switch_count={summary['test_switch_count']}, switch_ratio={_format_pct(summary['test_switch_ratio'])})")
    print(f"测试集可信率     : {_format_pct(summary['test_reliable_rate'])}")
    print(f"测试集可信正确率 : {_format_pct(summary['test_reliable_right_rate'])}")
    print(f"测试集 Loss      : {summary['test_loss']:.6f}")

    return summary


def _print_comparison_table(df):
    display_df = df.copy()
    for col in ['test_all_right_rate', 'test_avg_right_rate', 'test_switch_right_rate',
                'test_switch_ratio', 'test_reliable_rate', 'test_reliable_right_rate']:
        display_df[col] = display_df[col].map(_format_pct)
    display_df['test_loss'] = display_df['test_loss'].map(lambda x: f'{x:.6f}')

    print('\n' + '=' * 100)
    print('四种对比场景汇总')
    print('=' * 100)
    print(display_df[['label', 'test_all_right_rate', 'test_avg_right_rate',
                      'test_switch_right_rate', 'test_switch_count', 'test_switch_ratio',
                      'test_reliable_rate', 'test_reliable_right_rate', 'test_loss']].to_string(index=False))

    old_agg = df[df['case_name'] == 'old_agg_model_on_new_agg_data'].iloc[0]
    old_raw = df[df['case_name'] == 'old_raw_model_on_new_raw_data'].iloc[0]
    new_agg = df[df['case_name'] == 'new_agg_model_on_new_agg_data'].iloc[0]
    new_raw = df[df['case_name'] == 'new_raw_model_on_new_raw_data'].iloc[0]

    print('\n' + '=' * 100)
    print('关键差异（测试集平均正确率）')
    print('=' * 100)
    print(f"旧 aggregate 模型 vs 旧未 aggregate 模型 : {old_agg['test_avg_right_rate'] - old_raw['test_avg_right_rate']:.3f}%")
    print(f"新 aggregate 模型 vs 旧未 aggregate 模型 : {new_agg['test_avg_right_rate'] - old_raw['test_avg_right_rate']:.3f}%")
    print(f"新未 aggregate 模型 vs 旧未 aggregate 模型 : {new_raw['test_avg_right_rate'] - old_raw['test_avg_right_rate']:.3f}%")

    print('\n' + '=' * 100)
    print('关键差异（测试集切换子集准确率）')
    print('=' * 100)
    print(f"旧 aggregate 模型 vs 旧未 aggregate 模型 : {old_agg['test_switch_right_rate'] - old_raw['test_switch_right_rate']:.3f}%")
    print(f"新 aggregate 模型 vs 旧未 aggregate 模型 : {new_agg['test_switch_right_rate'] - old_raw['test_switch_right_rate']:.3f}%")
    print(f"新未 aggregate 模型 vs 旧未 aggregate 模型 : {new_raw['test_switch_right_rate'] - old_raw['test_switch_right_rate']:.3f}%")


def _save_summary(df):
    out_df = df.copy()
    out_df = out_df.rename(columns={
        'case_name': '场景代号',
        'label': '测试场景',
        'model_name': '模型名',
        'data_name': '数据名',
        'test_all_right_rate': '测试集全对率(%)',
        'test_avg_right_rate': '测试集平均正确率(%)',
        'test_switch_right_rate': '测试集切换子集准确率(%)',
        'test_switch_count': '测试集切换点数量',
        'test_switch_ratio': '测试集切换点占比(%)',
        'test_reliable_rate': '测试集可信率(%)',
        'test_reliable_right_rate': '测试集可信正确率(%)',
        'test_loss': '测试集Loss',
    })
    out_df.to_excel(SUMMARY_RESULT_XLSX, index=False)
    print(f"\n对比汇总已保存: {SUMMARY_RESULT_XLSX}")


if __name__ == '__main__':
    all_results = []
    for case_cfg in TEST_CASES:
        all_results.append(_run_single_case(case_cfg))

    result_df = pd.DataFrame(all_results)
    _print_comparison_table(result_df)
    _save_summary(result_df)
