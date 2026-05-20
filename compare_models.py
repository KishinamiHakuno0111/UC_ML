"""
DNN / Bi-LSTM / Transformer / Transformer-Aggregated 模型对比脚本
对比指标：训练集与测试集的平均准确率、可信率、可信正确率
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_basic_functions import test_model, get_data, ml_load_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================
#                    配置
# ============================================================
DATA_DICT = 'training_data'
MODEL_DICT = 'module_save'
RANDOM_SEED = 42
VAL_RATIO = 0.1
SAVE_XLSX = 'model_comparison_result.xlsx'

MODEL_CONFIGS = [
    {
        'label': 'DNN',
        'model_name': 'units_DNN_all_training_data_case1354pegase_storage_data_case1354pegase_0%.npz_random_1000_50000_[256, 256, 256]_0.0002',
    },
    {
        'label': 'Bi-LSTM',
        'model_name': 'units_BiLSTM_all_training_data_case1888rte_pro.npz_random_1000_50000_128_0.0002',
    },
    {
        'label': 'Transformer',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro.npz_random_1000_50000_128_0.0002',
    },
    {
        'label': 'Transformer-Aggregated',
        'model_name': 'units_transformer_all_training_data_case1888rte_pro_decreased_base.npz_random_1000_50000_128_0.0002',
    },
]


def main():
    # 清除旧结果文件，避免残留数据
    if os.path.exists(SAVE_XLSX):
        os.remove(SAVE_XLSX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    print(f"Device: {device}\n")

    # ============================================================
    #  逐模型加载、评估
    # ============================================================
    for cfg in MODEL_CONFIGS:
        print('=' * 80)
        print(f">>> 评估模型: {cfg['label']}")
        print(f"    model_name: {cfg['model_name']}")

        # 1) 加载模型及元信息
        model, model_type, training_num, test_num, \
            data_is_random, cutted_by_time, data_name, data_type = \
            ml_load_model(cfg['model_name'], MODEL_DICT)

        recognize_constr = (data_type == 'constraints')
        data_path = os.path.join(DATA_DICT, data_name)
        print(f"    model_type={model_type}, data={data_name}, "
              f"train={training_num}, test={test_num}, "
              f"random={data_is_random}, cutted_by_time={cutted_by_time}")

        # 2) 加载数据
        (train_inputs, train_targets,
         val_inputs, val_targets,
         test_inputs, test_targets,
         time_num, input_size, output_size,
         average_load, train_size) = get_data(
            model_type, data_path,
            test_num=test_num,
            training_num=training_num,
            data_is_random=data_is_random,
            cutted_by_time=cutted_by_time,
            recognize_constr=recognize_constr,
            random_seed=RANDOM_SEED,
            val_ratio=VAL_RATIO,
        )
        print(f"    train={len(train_inputs)}, val={len(val_inputs)}, test={len(test_inputs)}")

        # 3) 移到 device
        model = model.to(device)
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)

        # 4) 调用 test_model 评估（内部会打印详细结果并写入 Excel）
        _, _ = test_model(
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
            on_training=False,
            recognize_constr=recognize_constr,
            save_name=SAVE_XLSX,
            model_name=cfg['label'],
        )
        print()

    # ============================================================
    #  汇总对比
    # ============================================================
    df = pd.read_excel(SAVE_XLSX)

    # 提取对比所需列
    compare_cols = [
        'model_name',
        '训练集平均正确率', '测试集平均正确率',
        '训练集可信率',     '测试集可信率',
        '训练集可信正确率', '测试集可信正确率',
    ]
    df_compare = df[compare_cols].copy()
    df_compare = df_compare.rename(columns={'model_name': '模型'})

    # 格式化显示
    pct_cols = [c for c in df_compare.columns if c != '模型']
    df_display = df_compare.copy()
    for c in pct_cols:
        df_display[c] = df_display[c].map(lambda x: f'{x:.3f}%')

    print('\n' + '=' * 100)
    print('模型对比汇总（训练集 & 测试集）')
    print('=' * 100)
    print(df_display.to_string(index=False))
    print('=' * 100)

    # ============================================================
    #  可视化对比
    # ============================================================
    models = df_compare['模型'].tolist()
    metrics = ['平均正确率', '可信率', '可信正确率']
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        train_vals = df_compare[f'训练集{metric}'].values
        test_vals = df_compare[f'测试集{metric}'].values

        bars1 = ax.bar(x - width / 2, train_vals, width, label='Train', color='steelblue')
        bars2 = ax.bar(x + width / 2, test_vals, width, label='Test', color='coral')

        ax.set_ylabel('%')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('DNN / Bi-LSTM / Transformer / Transformer-Aggregated Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print(f"\n对比图已保存: model_comparison.png")
    print(f"对比数据已保存: {SAVE_XLSX}")
    plt.show()


if __name__ == '__main__':
    main()
