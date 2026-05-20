import json as _json
import numpy
import scipy.io
import torch
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime
import math

from ml_module_define import DNN, TransformerModel, BiLSTM_new, BiLSTM, KAN



def _replace_ext(path, new_ext):
    base, _ = os.path.splitext(path)
    return base + new_ext


def _load_array_dict(data_path):
    data_path = os.fspath(data_path)
    lower_path = data_path.lower()

    if lower_path.endswith('.npz'):
        npz_path = data_path
    elif lower_path.endswith('.mat'):
        npz_path = data_path[:-4] + '.npz'
    else:
        npz_path = data_path + '.npz'

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} does not exist. Only .npz training data is supported.")
    return dict(np.load(npz_path, allow_pickle=True)), 'npz'


# 随机生成训练数据（支持 train/val/test 三划分，防止过拟合）
def get_data(module_name, training_data, test_num, training_num, data_is_random=False,
             cutted_by_time=False, recognize_constr=False, random_seed=None,
             val_ratio=0.1, val_data=None):
    """
    新增参数:
        val_ratio  : 从 training_num 中划分多少比例作为验证集（默认 10%）
        val_data   : 若非 None，则使用外部传入的验证集；否则从 training_num 末尾抽取
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    learning_data, data_format = _load_array_dict(training_data)
    input_data = learning_data["input_data"]
    if data_format == 'mat':
        input_data = np.transpose(input_data, (0, 2, 1))
    input_data = input_data[0:test_num + training_num]

    if recognize_constr:
        target_data = learning_data["enable_data"]
        target_data = target_data[0:test_num + training_num]
        sample_num, l_num = target_data.shape
    else:
        target_data = learning_data["output_data"]
        target_data = target_data[0:test_num + training_num]
        if data_format == 'mat':
            sample_num, gen_num, time_num = target_data.shape
            target_data = np.transpose(target_data, (0, 2, 1))
        else:
            sample_num, time_num, gen_num = target_data.shape

    # input_data is now (sample, time, bus) for both formats
    sample_num, time_num, bus_num = input_data.shape
    average_load = np.asarray(learning_data['average_load']).item()

    if data_is_random:
        indices = list(range(len(input_data)))
        random.shuffle(indices)
        input_data = np.array([input_data[i] for i in indices])
        target_data = np.array([target_data[i] for i in indices])

    target_data = target_data.astype(np.float32)

    # ---------- 三集合划分 ----------
    # 数据排列顺序：[train, val, test]。先按完整时序样本切分，再为 DNN/KAN 展开时间维度。
    if val_data is None:
        val_size = int(training_num * val_ratio)
        train_size = training_num - val_size
        train_inputs = input_data[0:train_size]
        train_targets = target_data[0:train_size]
        val_inputs = input_data[train_size:training_num]
        val_targets = target_data[train_size:training_num]
        test_inputs = input_data[training_num:training_num + test_num]
        test_targets = target_data[training_num:training_num + test_num]
    else:
        train_inputs = input_data[0:training_num]
        train_targets = target_data[0:training_num]
        val_inputs, val_targets = val_data
        test_inputs = input_data[training_num:training_num + test_num]
        test_targets = target_data[training_num:training_num + test_num]

    if module_name in ('DNN', 'KAN'):
        if cutted_by_time:
            train_inputs = np.reshape(train_inputs, [-1, bus_num])
            val_inputs = np.reshape(val_inputs, [-1, bus_num])
            test_inputs = np.reshape(test_inputs, [-1, bus_num])
            input_size = bus_num
            if recognize_constr:
                output_size = l_num
            else:
                train_targets = np.reshape(train_targets, [-1, gen_num])
                val_targets = np.reshape(val_targets, [-1, gen_num])
                test_targets = np.reshape(test_targets, [-1, gen_num])
                output_size = gen_num
        else:
            train_inputs = np.reshape(train_inputs, [len(train_inputs), bus_num * time_num])
            val_inputs = np.reshape(val_inputs, [len(val_inputs), bus_num * time_num])
            test_inputs = np.reshape(test_inputs, [len(test_inputs), bus_num * time_num])
            input_size = bus_num * time_num
            if recognize_constr:
                output_size = l_num
            else:
                train_targets = np.reshape(train_targets, [len(train_targets), gen_num * time_num])
                val_targets = np.reshape(val_targets, [len(val_targets), gen_num * time_num])
                test_targets = np.reshape(test_targets, [len(test_targets), gen_num * time_num])
                output_size = gen_num * time_num
    else:
        input_size = bus_num
        output_size = l_num if recognize_constr else gen_num

    train_inputs_t = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets_t = torch.tensor(train_targets, dtype=torch.float32)
    val_inputs_t = torch.tensor(val_inputs, dtype=torch.float32)
    val_targets_t = torch.tensor(val_targets, dtype=torch.float32)
    test_inputs_t = torch.tensor(test_inputs, dtype=torch.float32)
    test_targets_t = torch.tensor(test_targets, dtype=torch.float32)

    train_size = len(train_inputs_t)

    return (train_inputs_t, train_targets_t,
            val_inputs_t, val_targets_t,
            test_inputs_t, test_targets_t,
            time_num, input_size, output_size, average_load, train_size)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_ratio=0.0, last_epoch=-1):
    """
    Warmup + Cosine Annealing 学习率调度器。
    调度曲线：
      - 0 ~ num_warmup_steps：线性 warmup (lr=0 → base_lr)
      - num_warmup_steps ~ num_training_steps：余弦退火 (base_lr → base_lr * min_lr_ratio)
    """
    base_lr = optimizer.defaults['lr']
    warmup_factor = 1.0 / num_warmup_steps if num_warmup_steps > 0 else 1.0
    min_lr = base_lr * min_lr_ratio

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) * warmup_factor
        progress = float(current_step - num_warmup_steps) / float(max(1, current_step - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, (cosine_decay * (1.0 - warmup_factor) + warmup_factor))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def training(model, criterion, num_epochs, inputs, targets, val_inputs, val_targets,
             learning_rate, weight_decay, device, stop_gap,
             use_scheduler=True, warmup_ratio=0.1,
             gradient_clip_val=1.0, accumulate_steps=1,
             early_stopping_patience=50, early_stopping_metric='val_loss',
             checkpoint_path=None, logger=print):
    """
    改进的训练函数。

    新增参数：
        val_inputs / val_targets : 验证集（用于 Early Stopping 和调度器）
        use_scheduler            : 是否使用 warmup + cosine annealing 调度
        warmup_ratio             : warmup 占总步数的比例（默认 10%）
        gradient_clip_val        : 梯度裁剪阈值（0 表示不裁剪）
        accumulate_steps         : 梯度累积步数，用于模拟大 batch
        early_stopping_patience  : Early Stopping 耐心值（验证 loss 多少轮没改善就停止）
        early_stopping_metric    : 早停指标，'val_loss' 或 'train_loss'
        checkpoint_path          : 断点续训路径（保存最佳模型）
        logger                   : 日志输出函数
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = (num_epochs // stop_gap) * accumulate_steps
    num_warmup_steps = int(total_steps * warmup_ratio)

    if use_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, total_steps, min_lr_ratio=0.01)
    else:
        scheduler = None

    batch_size = len(inputs)  # 每步使用全部训练数据（可改为更小的 batch）

    loss_record = []
    val_loss_record = []

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        # ---------- 梯度累积：每 accumulate_steps 个子步做一次参数更新 ----------
        optimizer.zero_grad()
        for sub_step in range(accumulate_steps):
            step_input = inputs
            step_target = targets
            output = model(step_input)
            loss = criterion(output, step_target)
            (loss / accumulate_steps).backward()

        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ---------- 定期评估 ----------
        if (epoch + 1) % stop_gap == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger(f"Epoch [{epoch + 1}/{num_epochs}], LR={current_lr:.2e}", end='')

            # 训练 loss
            model.eval()
            with torch.no_grad():
                train_out = model(inputs)
                train_loss = criterion(train_out, targets).item()

                if val_inputs is not None:
                    val_out = model(val_inputs)
                    val_loss = criterion(val_out, val_targets).item()
                    logger(f", TrainLoss={train_loss:.6f}, ValLoss={val_loss:.6f}", end='')
                else:
                    val_loss = train_loss

            loss_record.append(train_loss)
            val_loss_record.append(val_loss)

            # ---------- Early Stopping ----------
            if early_stopping_metric == 'val_loss':
                monitor_loss = val_loss if val_inputs is not None else train_loss
            else:
                monitor_loss = train_loss

            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_path:
                    torch.save(best_model_state, checkpoint_path + '.best.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger(f"\n[EarlyStopping] 验证集 loss {early_stopping_patience} 轮未改善，停止训练。最佳 epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                    break

            logger("")

        if (epoch + 1) % 1000 == 0:
            fig = plt.figure(figsize=(10, 4))
            plt.plot(list(range(stop_gap, len(loss_record) * stop_gap + 1, stop_gap)), loss_record, label='Train')
            if val_loss_record:
                plt.plot(list(range(stop_gap, len(val_loss_record) * stop_gap + 1, stop_gap)), val_loss_record, label='Val')
            plt.title('Loss Curve')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.close(fig)

    # ---------- 恢复最佳模型 ----------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger(f"已恢复最佳模型（epoch={best_epoch}），ValLoss={best_val_loss:.6f}")

    return model, loss_record, val_loss_record


def test_model(model, criterion, device, loss_record, val_loss_record, inputs, targets,
              test_inputs, test_targets, val_inputs=None, val_targets=None,
              stop_gap=50, reliability_low_gate=0.1, reliability_high_gate=0.9,
              on_training=True, recognize_constr=False, save_name='test_output.xlsx',
              model_name=''):
    model.eval()  # 改为测试模式

    device_cpu = 'cpu'
    training_num = len(inputs)
    val_num = len(val_inputs) if val_inputs is not None else 0
    test_num = len(test_inputs)

    def _eval_split(split_inputs, split_targets, split_name):
        split_input_dev = split_inputs.to(device)
        split_target_dev = split_targets.to(device)
        split_output = model(split_input_dev)
        split_loss = criterion(split_output, split_targets.to(device)).item()

        split_output_cpu = split_output.to(device_cpu).detach().numpy()
        split_target_np = split_targets.to(device_cpu).detach().numpy()

        delta = abs(split_output_cpu - split_target_np)
        check_right = (delta < 0.5)
        n_dim = check_right.ndim
        all_num = check_right.size

        if n_dim == 2:
            unit_num = check_right.shape[1]
            true_num_matrix = np.sum(check_right, axis=1)
            all_right_num = sum(abs(unit_num - true_num_matrix) < 1e-3)
            all_true_num = np.sum(check_right)
            zero_wrong_delta = split_output_cpu - split_target_np
            zero_wrong_check = (zero_wrong_delta > 0.5)
            zero_wrong_num = np.sum(zero_wrong_check)
        else:
            unit_num = check_right.shape[1] * check_right.shape[2]
            true_num_matrix = np.sum(check_right, axis=(1, 2))
            all_right_num = sum(abs(unit_num - true_num_matrix) < 1e-3)
            all_true_num = np.sum(check_right)
            zero_wrong_delta = split_output_cpu - split_target_np
            zero_wrong_check = (zero_wrong_delta > 0.5)
            zero_wrong_num = np.sum(zero_wrong_check)

        all_wrong_num = all_num - all_true_num
        one_wrong_num = all_wrong_num - zero_wrong_num

        is_reliable = (split_output_cpu <= reliability_low_gate) + (split_output_cpu >= reliability_high_gate)
        all_reliable_num = np.sum(is_reliable)
        all_reliable_true_num = np.sum(is_reliable * check_right)

        split_size = len(split_inputs)

        print(f"{split_name}集全对率为{all_right_num * 100 / split_size:.2f}%，"
              f"平均正确率为{all_true_num * 100 / all_num:.2f}%，"
              f"可信率为{100*all_reliable_num/all_num:.2f}%，"
              f"可信正确率为{100*all_reliable_true_num/all_reliable_num:.2f}%" if all_reliable_num > 0 else
              f"{split_name}集全对率为{all_right_num * 100 / split_size:.2f}%，"
              f"平均正确率为{all_true_num * 100 / all_num:.2f}%。")
        print(f"{split_name}集错误率为{all_wrong_num*100/all_num:.2f}%，"
              f"0错认为1为{zero_wrong_num*100/all_num:.2f}%，"
              f"1错认为0为{one_wrong_num*100/all_num:.2f}%，Loss={split_loss:.6f}")

        return {
            'all_right_rate': all_right_num * 100 / split_size,
            'avg_right_rate': all_true_num * 100 / all_num,
            'reliable_rate': 100 * all_reliable_num / all_num,
            'reliable_right_rate': 100 * all_reliable_true_num / all_reliable_num if all_reliable_num > 0 else 0,
            'wrong_rate': all_wrong_num * 100 / all_num,
            'zero_wrong_rate': zero_wrong_num * 100 / all_num,
            'one_wrong_rate': one_wrong_num * 100 / all_num,
            'loss': split_loss,
            'all_reliable_num': all_reliable_num,
            'all_num': all_num,
        }

    print("=" * 60)
    print("进行训练集测试")
    train_result = _eval_split(inputs, targets, "训练")

    print("=" * 60)
    val_result = None
    if val_inputs is not None and len(val_inputs) > 0:
        print("进行验证集测试")
        val_result = _eval_split(val_inputs, val_targets, "验证")

    print("=" * 60)
    print("进行测试集测试")
    test_result = _eval_split(test_inputs, test_targets, "测试")

    # 写入测试数据
    record = {
        'time': [datetime.now().strftime('%Y年%m月%d日 %H:%M')],
        'model_name': [model_name],
        '训练集全对率': [train_result['all_right_rate']],
        '验证集全对率': [val_result['all_right_rate'] if val_result else None],
        '测试集全对率': [test_result['all_right_rate']],
        '训练集平均正确率': [train_result['avg_right_rate']],
        '验证集平均正确率': [val_result['avg_right_rate'] if val_result else None],
        '测试集平均正确率': [test_result['avg_right_rate']],
        '训练集可信率': [train_result['reliable_rate']],
        '验证集可信率': [val_result['reliable_rate'] if val_result else None],
        '测试集可信率': [test_result['reliable_rate']],
        '训练集可信正确率': [train_result['reliable_right_rate']],
        '验证集可信正确率': [val_result['reliable_right_rate'] if val_result else None],
        '测试集可信正确率': [test_result['reliable_right_rate']],
        '训练集Loss': [train_result['loss']],
        '验证集Loss': [val_result['loss'] if val_result else None],
        '测试集Loss': [test_result['loss']],
    }

    df = pd.DataFrame(record)

    if os.path.exists(save_name):
        df_existing = pd.read_excel(save_name)
        if list(df.columns) == list(df_existing.columns):
            df = pd.concat([df_existing, df], ignore_index=True)

    df.to_excel(save_name, index=False)

    if on_training:
        plt.figure(figsize=(12, 4))
        iterations = list(range(stop_gap, len(loss_record) * stop_gap + 1, stop_gap))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, loss_record, label='Train', color='steelblue')
        if val_loss_record:
            plt.plot(iterations, val_loss_record, label='Val', color='coral')
        plt.title('Loss Curve')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        all_reliable_rates = [
            train_result['reliable_rate'],
            val_result['reliable_rate'] if val_result else 0,
            test_result['reliable_rate']
        ]
        all_right_rates = [
            train_result['all_right_rate'],
            val_result['all_right_rate'] if val_result else 0,
            test_result['all_right_rate']
        ]
        x = np.arange(3)
        width = 0.35
        bars1 = plt.bar(x - width/2, all_reliable_rates, width, label='Acc@Reliable', color='steelblue')
        bars2 = plt.bar(x + width/2, all_right_rates, width, label='All-Right Rate', color='coral')
        plt.xticks(x, ['Train', 'Val', 'Test'])
        plt.ylabel('%')
        plt.title('Accuracy Metrics')
        plt.legend()
        for bar in bars1:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt_result = plt
        return plt_result, test_result
    else:
        return 0, test_result


def test_output_decreased_model(model, criterion, device, loss_record, inputs, targets, test_inputs, test_targets, case_save, stop_gap = 50, num_epochs = 10000, reliability_low_gate = 0.1, reliability_high_gate=0.9, on_training = True, recognize_constr=False):
    model.eval()  # 改为测试模式

    train_right_num = 0
    device_cpu = 'cpu'
    device_cuda = 'cuda'

    #  进行训练集测试
    print("进行训练集测试")

    ture_output = targets
    train_input = inputs.to(device)
    ture_output = ture_output.to(device)
    train_output = model(train_input)
    train_loss = criterion(train_output, ture_output)

    train_output = train_output.to(device_cpu)
    ture_output = ture_output.to(device_cpu)
    train_output = train_output.detach().numpy()
    ture_output = ture_output.detach().numpy()

    train_num, time_num, output_size = train_output.shape
    # 获得每次的预测结果
    train_prediction = np.argmax(train_output, axis=2)
    train_ture_prediction = np.argmax(ture_output, axis=2)
    delta = abs(train_prediction - train_ture_prediction)
    check_right = (delta < 0.5)
    all_train_ture_num = check_right.size




    #  进行测试集测试
    print("进行测试集测试")
    ture_output = test_targets
    test_input = test_inputs.to(device)
    ture_output = ture_output.to(device)
    test_output = model(test_input)
    test_loss = criterion(test_output, ture_output)

    test_output = test_output.to(device_cpu)
    ture_output = ture_output.to(device_cpu)
    test_output = test_output.detach().numpy()
    ture_output = ture_output.detach().numpy()

    # 获取预测为1，但实际为0的值
    zero_wrong_delta = test_output - ture_output
    zero_wrong_check = (zero_wrong_delta > 0.5)
    n_dim = zero_wrong_check.ndim
    if n_dim == 2:
        test_zero_wrong_num = np.sum(zero_wrong_check, axis=(0, 1))
    elif n_dim == 3:
        test_zero_wrong_num = np.sum(zero_wrong_check, axis=(0, 1, 2))

    is_reliable = (test_output <= reliability_low_gate) + (test_output >= reliability_high_gate)

    delta = abs(test_output - ture_output)
    check_right = (delta < 0.5)
    all_test_bus_time_num = check_right.size

    if n_dim == 2:
        bus_time_num = check_right.shape[1]
        true_num_matrix = np.sum(check_right, axis=1)
        forecast_is_true = abs(bus_time_num - true_num_matrix) < 1e-3
        test_right_num = sum(forecast_is_true)
        test_all_true_num = np.sum(check_right, axis=(0, 1))
        test_wrong_num = test_num - test_right_num

        test_all_reliable_num = np.sum(is_reliable, axis=(0, 1))
        test_all_reliable_true_num = np.sum(is_reliable * check_right, axis=(0, 1))
    elif n_dim == 3:
        bus_time_num = check_right.shape[1] * check_right.shape[2]
        true_num_matrix = np.sum(check_right, axis=(1, 2))
        forecast_is_true = abs(bus_time_num - true_num_matrix) < 1e-3
        test_right_num = sum(forecast_is_true)
        test_all_true_num = np.sum(check_right, axis=(0, 1, 2))
        test_wrong_num = test_num - test_right_num

        test_all_reliable_num = np.sum(is_reliable, axis=(0, 1, 2))
        test_all_reliable_true_num = np.sum(is_reliable * check_right, axis=(0, 1, 2))

    # 预测为0，但实际为1的数量
    test_all_wrong_num = all_test_bus_time_num - test_all_true_num
    test_one_wrong_num = test_all_wrong_num - test_zero_wrong_num

    print(
        f"训练集平均正确率为{train_right_num * 100 / training_num}%，测试集平均正确率为{test_right_num * 100 / test_num}%")
    print(
        f"从每个机组的正确率看，训练集平均正确率为{train_all_true_num * 100 / all_train_bus_time_num}%，测试集平均正确率为{test_all_true_num * 100 / all_test_bus_time_num}%")
    print(
        f"只看通过可信范围门槛的，训练集中有{100 * train_all_reliable_num / all_train_bus_time_num}%的数据可信，其中正确率为{100 * train_all_reliable_true_num / train_all_reliable_num}%"
        f"测试集中有{100 * test_all_reliable_num / all_test_bus_time_num}%的数据可信，其中正确率为{100 * test_all_reliable_true_num / test_all_reliable_num}%")
    print(
        f"训练集的错误率为{train_all_wrong_num * 100 / all_train_bus_time_num}%，其中预测为1，但实际为0的为{train_zero_wrong_num * 100 / all_train_bus_time_num}%, 预测为0， 实际为1的为{train_one_wrong_num * 100 / all_train_bus_time_num}%")
    print(
        f"测试集的错误率为{test_all_wrong_num * 100 / all_test_bus_time_num}%，其中预测为1，但实际为0的为{test_zero_wrong_num * 100 / all_test_bus_time_num}%, 预测为0， 实际为1的为{test_one_wrong_num * 100 / all_test_bus_time_num}%")

    print(f"训练集的loss为{train_loss}, 测试集的loss为{test_loss}")
    if on_training:
        plt.plot(list(range(stop_gap, num_epochs + 1, stop_gap)), loss_record)
        plt.show()

def _mat_to_py(val):
    """Convert scipy scalar/array values to plain Python types."""
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        if val.ndim == 1:
            return val.tolist()
        if val.ndim == 2 and val.shape[0] == 1:
            return val[0].tolist()
        if val.ndim == 2 and val.shape[1] == 1:
            return val[:, 0].tolist()
    return val


def ml_load_model(model_name, dict_name):
    model_file_path = os.path.join(dict_name, model_name + '.pth')
    model = torch.load(model_file_path, weights_only=False)

    # Prefer JSON sidecar (new format), fall back to .mat
    json_path = os.path.join(dict_name, model_name + '_metadata.json')
    mat_path  = os.path.join(dict_name, model_name + '.mat')

    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            meta = _json.load(f)
        model_type    = meta.get('model_type',    meta.get('model_type',    'unknown'))
        training_num  = int(meta.get('training_num',  0))
        test_num      = int(meta.get('test_num',      0))
        data_is_random = bool(meta.get('data_is_random', False))
        cutted_by_time = bool(meta.get('cutted_by_time', False))
        data_name      = str(meta.get('data_name',     ''))
        data_type      = str(meta.get('data_type',     ''))
    elif os.path.exists(mat_path):
        model_data = scipy.io.loadmat(mat_path)
        model_type    = _mat_to_py(model_data['model_type'])
        training_num  = int(_mat_to_py(model_data['training_num']))
        test_num      = int(_mat_to_py(model_data['test_num']))
        data_is_random = bool(_mat_to_py(model_data['data_is_random']))
        cutted_by_time = bool(_mat_to_py(model_data['cutted_by_time']))
        data_name      = str(_mat_to_py(model_data['data_name']))
        data_type      = str(_mat_to_py(model_data['data_type']))
    else:
        raise FileNotFoundError(
            f"Neither {json_path} nor {mat_path} found for model '{model_name}'"
        )

    return model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type

def train_model(module_name, inputs, targets, test_inputs, test_targets, input_size, output_size, modulel_path,
                 hidden_sizes=64, learning_rate=0.001, num_epochs=10000, dropout_prob=0.3, num_pieces=2,
                 is_maxout=False, stop_gap=50, weight_decay=0.001, recognize_constr=False,
                 val_inputs=None, val_targets=None,
                 use_scheduler=True, warmup_ratio=0.1,
                 gradient_clip_val=1.0, accumulate_steps=1,
                 early_stopping_patience=50, early_stopping_metric='val_loss',
                 reliability_low_gate=0.1, reliability_high_gate=0.9,
                 save_name='test_output.xlsx'):
    """
    改进的训练函数入口。

    新增参数：
        val_inputs / val_targets  : 验证集张量（支持 None，此时从训练集末尾自动划分）
        use_scheduler             : 是否启用 warmup + cosine annealing 学习率调度
        warmup_ratio             : warmup 步数占总步数的比例
        gradient_clip_val         : 梯度裁剪阈值（设为 0 则不裁剪）
        accumulate_steps          : 梯度累积步数（模拟大 batch）
        early_stopping_patience  : Early Stopping 耐心值
        early_stopping_metric    : 早停监控指标，'val_loss' 或 'train_loss'
        reliability_low_gate      : 可信度下限
        reliability_high_gate    : 可信度上限
        save_name                : 测试结果 Excel 保存路径
    """
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU 状态: {torch.cuda.is_available()}\n")

    # ---------- 自动从训练集末尾划分验证集 ----------
    if val_inputs is None:
        val_ratio = 0.1
        val_size = int(len(inputs) * val_ratio)
        train_size = len(inputs) - val_size
        _val_inputs = inputs[train_size:]
        _val_targets = targets[train_size:]
        _train_inputs = inputs[:train_size]
        _train_targets = targets[:train_size]
        print(f"[数据集划分] 训练 {train_size} | 验证 {val_size} | 测试 {len(test_inputs)}")
    else:
        _train_inputs = inputs
        _train_targets = targets
        _val_inputs = val_inputs
        _val_targets = val_targets
        print(f"[数据集划分] 训练 {len(inputs)} | 验证 {len(val_inputs)} | 测试 {len(test_inputs)}")

    # ---------- 移动到 GPU ----------
    _train_inputs = _train_inputs.to(device)
    _train_targets = _train_targets.to(device)
    _val_inputs = _val_inputs.to(device)
    _val_targets = _val_targets.to(device)
    test_inputs_dev = test_inputs.to(device)
    test_targets_dev = test_targets.to(device)

    # ---------- 模型定义 ----------
    if module_name == 'DNN':
        model = DNN(input_size, hidden_sizes, output_size, dropout_prob, num_pieces, is_maxout).to(device)
    elif module_name == 'KAN':
        model = KAN(input_size, hidden_sizes, output_size, dropout_prob, num_pieces, is_maxout).to(device)
    elif module_name == 'transformer':
        model = TransformerModel(input_size=input_size, output_size=output_size, hidden_dim=hidden_sizes).to(device)
    elif module_name == 'LSTM_new':
        model = BiLSTM_new(input_size, hidden_sizes, output_size).to(device)
    elif module_name == 'BiLSTM':
        model = BiLSTM(input_size, hidden_sizes, output_size, dropout_prob).to(device)

    # ---------- 训练 ----------
    model, loss_record, val_loss_record = training(
        model, criterion, num_epochs,
        _train_inputs, _train_targets,
        _val_inputs, _val_targets,
        learning_rate, weight_decay, device, stop_gap,
        use_scheduler=use_scheduler, warmup_ratio=warmup_ratio,
        gradient_clip_val=gradient_clip_val, accumulate_steps=accumulate_steps,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        checkpoint_path=modulel_path
    )
    print("训练完成")

    # ---------- 保存模型 ----------
    torch.save(model, modulel_path + '.pth')

    # ---------- 最终测试 ----------
    plt_result, test_result = test_model(
        model, criterion, device,
        loss_record, val_loss_record,
        _train_inputs, _train_targets,
        test_inputs_dev, test_targets_dev,
        val_inputs=_val_inputs, val_targets=_val_targets,
        stop_gap=stop_gap,
        reliability_low_gate=reliability_low_gate,
        reliability_high_gate=reliability_high_gate,
        on_training=True,
        recognize_constr=recognize_constr,
        save_name=save_name,
        model_name=modulel_path.split('\\')[-1]
    )

    # 保存 loss 数据
    iterations = list(range(stop_gap, len(loss_record) * stop_gap + 1, stop_gap))
    import pandas as pd
    df_loss = pd.DataFrame({'train_loss': loss_record, 'val_loss': val_loss_record, 'iteration': iterations})
    df_loss.to_csv(modulel_path + '_loss.csv', index=False)

    return model, loss_record, val_loss_record, plt_result, test_result

def magnify_peak_valley_difference(inputs, rate):
    n_dim = inputs.ndim
    if n_dim == 3:
        data_num, bus_num, time_num = inputs.shape
        average_load = np.sum(inputs, axis=1)/bus_num
        max_load = np.max(inputs, axis=1)
        min_load = np.min(inputs, axis=1)
        max_load_matrix = np.array([max_load for i in range(bus_num)])
        min_load_matrix = np.array([min_load for i in range(bus_num)])
        max_load_matrix = np.transpose(max_load_matrix, axes=(1, 0, 2))
        min_load_matrix = np.transpose(min_load_matrix, axes=(1, 0, 2))
        k = (rate * max_load_matrix - min_load_matrix/rate)/(max_load_matrix - min_load_matrix)
        new_inputs = min_load_matrix/rate + (inputs - min_load_matrix) * k
    elif n_dim == 2:
        bus_num, time_num = inputs.shape
        max_load = np.max(inputs, axis=0)
        min_load = np.min(inputs, axis=0)
        max_load_matrix = np.array([max_load for i in range(bus_num)])
        min_load_matrix = np.array([min_load for i in range(bus_num)])
        k = (rate * max_load_matrix - min_load_matrix/rate)/(max_load_matrix - min_load_matrix)
        new_inputs = min_load_matrix/rate + (inputs - min_load_matrix) * k
    else:
        print("Error!输入矩阵维度不正确！")
    return new_inputs

def try_get_from_list(x, t):
    return x[t] if type(x) == list else x

def get_load(data):
    bus_names, buses = zip(*data['Buses'].items())
    time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
    time_step = data['Parameters'].get('Time step (min)', 60)
    assert(60 % time_step == 0)
    time_multiplier = int(60 / time_step)
    NB = len(buses)
    NT = time_horizon * time_multiplier
    load_data = np.array([[try_get_from_list(buses[i]['Load (MW)'], t) for t in range(NT)] for i in range(NB)])
    return load_data
