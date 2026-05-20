import os
import json as _json
from ml_basic_functions import train_model, get_data
import pandas as pd
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================
#                    通用超参数配置
# ============================================================
# 数据配置
data_dict_name = 'training_data'
training_data = 'all_training_data_case118_storage_data_case118_30%'
training_path = data_dict_name + '\\' + training_data

training_num = 1000    # 训练样本总数（不含验证集划分）
test_num = 100         # 测试样本数
data_is_random = True   # 是否随机打乱数据
val_ratio = 0.1        # 验证集占总训练样本比例（新增）
random_seed = 42       # 随机种子，便于复现

# 训练配置
num_epochs = 50000     # 最大训练轮数（会被 Early Stopping 中断）
stop_gap = 50         # 每隔多少轮评估一次并记录 loss

# ============================================================
#                    模型选择
# ============================================================
# 可选: 'DNN', 'KAN', 'transformer', 'LSTM_new', 'BiLSTM'
module_list = ['DNN', 'KAN', 'transformer', 'LSTM_new', 'BiLSTM']
module_name = module_list[4]   # 0=DNN, 1=KAN, 2=Transformer, 3=LSTM_new, 4=BiLSTM

recognize_constraints = False  # True=训练约束辨识模型，False=训练机组启停预测模型

# ============================================================
#                    优化器配置
# ============================================================
learning_rate = 0.0002        # 初始学习率
weight_decay = 0.001          # L2 正则化权重
use_scheduler = True           # 是否启用 warmup + cosine annealing 学习率调度
warmup_ratio = 0.1            # warmup 步数占总步数的比例

# ============================================================
#                    训练稳定性配置
# ============================================================
gradient_clip_val = 1.0       # 梯度裁剪阈值（设为 0 则不裁剪）
accumulate_steps = 1          # 梯度累积步数（可模拟更大 batch）
early_stopping_patience = 50   # 验证集 loss 连续多少轮未改善则停止
early_stopping_metric = 'val_loss'  # 'val_loss' 或 'train_loss'

# ============================================================
#                    模型结构配置
# ============================================================
is_maxout = False
num_pieces = 2
dropout_prob = 0.3

if module_name == 'DNN':
    hidden_sizes = [256, 256, 256]
elif module_name == 'KAN':
    hidden_sizes = [256, 256, 256]
elif module_name == 'transformer':
    hidden_sizes = 128
elif module_name == 'LSTM_new':
    hidden_sizes = [64 for _ in range(36)]
elif module_name == 'BiLSTM':
    hidden_sizes = 128

# ============================================================
#                    可信度阈值配置
# ============================================================
reliability_low_gate = 0.1   # 可信度下限
reliability_high_gate = 0.9   # 可信度上限

# ============================================================
#                    输出路径配置
# ============================================================
dict_name = 'module_save'
fig_dict_name = 'fig_data'
os.makedirs(dict_name, exist_ok=True)
os.makedirs(fig_dict_name, exist_ok=True)
save_name = 'test_result.xlsx'

# ============================================================
#                    数据切分方式配置
# ============================================================
if recognize_constraints:
    cutted_by_time = False
else:
    if module_name == 'DNN':
        cutted_by_time = True   # DNN: 打散时间维度，视为独立样本
    elif module_name in ('transformer', 'BiLSTM'):
        cutted_by_time = False  # Transformer/BiLSTM: 保持时序结构
    else:
        cutted_by_time = False

# ============================================================
#                    生成文件保存路径
# ============================================================
if data_is_random:
    shuffle_str = 'random'
else:
    shuffle_str = 'not_random'

data_type = 'constraints' if recognize_constraints else 'units'
modulel_path = (f'{dict_name}\\{data_type}_{module_name}_{training_data}_{shuffle_str}'
                f'_{training_num}_{num_epochs}_{hidden_sizes}_{learning_rate}')

# ============================================================
#                    加载数据（支持 train/val/test 三划分）
# ============================================================
print("正在加载数据...")
(train_inputs, train_targets,
 val_inputs, val_targets,
 test_inputs, test_targets,
 time_num, input_size, output_size, average_load,
 train_size) = get_data(
    module_name, training_path,
    test_num=test_num,
    training_num=training_num,
    data_is_random=data_is_random,
    cutted_by_time=cutted_by_time,
    recognize_constr=recognize_constraints,
    random_seed=random_seed,
    val_ratio=val_ratio
)

print(f"数据维度: input={input_size}, output={output_size}, time_num={time_num}")
print(f"样本数: train={train_size}, val={len(val_inputs)}, test={len(test_inputs)}")

# ============================================================
#                    存储训练元信息
# ============================================================
meta = {
    'model_type': module_name,
    'training_num': training_num,
    'test_num': test_num,
    'val_ratio': val_ratio,
    'data_is_random': data_is_random,
    'cutted_by_time': cutted_by_time,
    'data_name': training_data,
    'data_type': data_type,
    'use_scheduler': use_scheduler,
    'warmup_ratio': warmup_ratio,
    'early_stopping_patience': early_stopping_patience,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
}
with open(modulel_path + '_metadata.json', 'w', encoding='utf-8') as f:
    _json.dump(meta, f, indent=2)

# ============================================================
#                    开始训练
# ============================================================
print(f"\n{'='*60}")
print(f"模型: {module_name} | 数据: {training_data}")
print(f"训练样本: {train_size} | 验证样本: {len(val_inputs)} | 测试样本: {len(test_inputs)}")
print(f"学习率: {learning_rate} (scheduler={'warmup+cosine' if use_scheduler else 'fixed'})")
print(f"Early Stopping: patience={early_stopping_patience}, metric={early_stopping_metric}")
print(f"{'='*60}\n")

train_start_time = time.time()
model, loss_record, val_loss_record, plt_result, test_result = train_model(
    module_name,
    train_inputs, train_targets,
    test_inputs, test_targets,
    input_size, output_size,
    modulel_path,
    hidden_sizes=hidden_sizes,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    dropout_prob=dropout_prob,
    num_pieces=num_pieces,
    is_maxout=is_maxout,
    stop_gap=stop_gap,
    weight_decay=weight_decay,
    recognize_constr=recognize_constraints,
    val_inputs=val_inputs,
    val_targets=val_targets,
    use_scheduler=use_scheduler,
    warmup_ratio=warmup_ratio,
    gradient_clip_val=gradient_clip_val,
    accumulate_steps=accumulate_steps,
    early_stopping_patience=early_stopping_patience,
    early_stopping_metric=early_stopping_metric,
    reliability_low_gate=reliability_low_gate,
    reliability_high_gate=reliability_high_gate,
    save_name=save_name
)

train_end_time = time.time()
print(f"\n训练总耗时: {train_end_time - train_start_time:.1f} 秒")

# ============================================================
#                    保存训练曲线图
# ============================================================
plt_result.savefig(modulel_path + '_curve.jpeg', format='jpeg', dpi=150)
print(f"训练曲线已保存: {modulel_path}_curve.jpeg")
plt_result.show()
