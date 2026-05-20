"""
generate_new_input_data.py
=========================
在原有数据基础上通过优化求解生成新的数据，用于测试模型迁移能力。
核心理念：模拟电力系统发展场景——新增新能源装机（zero bus 上加负负荷），
原有节点负荷可增可减。所有新增扰动的净注入之和不为零（远离零）。

核心设计：
- zero bus 新增负荷为负数（新能源作为负负荷）
- nonzero bus 只增加正负荷，用于抵消 zero bus 新能源导致的净负荷下降
- 只保留输出切换复杂度没有明显下降的样本（min_switch_complexity_ratio 过滤）
- 输出与输入同步更新，不存在历史数据污染
"""

import os
import gzip
import json
import numpy as np
import torch
import pyomo.environ as pyo
from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


# ── 可调参数 ────────────────────────────────────────────────────────────────

#  扰动幅度（相对于平均负荷的比例）
PERTURB_ZERO_RATIO  = 2.0   # zero bus: 扰动 = -P * average_load  （负 = 新能源装机）
PERTURB_NZ_RATIO    = 1.5   # nonzero bus: 每个时段新增正负荷上限 = P * average_load
LOAD_COMPENSATION_RATIO = 1.0  # 新增正负荷抵消 zero bus 新能源的比例
MIN_SWITCH_COMPLEXITY_RATIO = 0.8  # 新输出切换次数不能低于 base 太多
SAVE_INTERVAL       = 50    # 每生成多少样本保存一次

# ── 内部辅助 ────────────────────────────────────────────────────────────────


def _load_array(path):
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io
    return scipy.io.loadmat(path)


def get_history_data(dict_name, data_name):
    d = _load_array(os.path.join(dict_name, data_name))
    for key in ('input_data', 'output_data', 'l_data'):
        if key in d and d[key].ndim == 3:
            d[key] = np.transpose(d[key], (0, 2, 1))
    return d['input_data'], d['output_data'], d['enable_data'], d['l_data']


def _gen_perturbation(input_load_i, average_load, perturb_zero, perturb_nz,
                      load_compensation_ratio=LOAD_COMPENSATION_RATIO):
    """
    为单个样本生成输入扰动。

    策略：模拟新能源装机扩张
    - zero bus（无负荷节点）：加负扰动（新能源作为负负荷）
    - nonzero bus：只加随机正负荷，用于补偿 zero bus 新能源

    参数
    ----
    input_load_i : (B, T)  MW 单位的负荷数据
    average_load : float    平均负荷（MW）
    perturb_zero : float    zero bus 的扰动幅度上限（正值，最终取负）
    perturb_nz   : float    nonzero bus 每个时段新增正负荷上限

    返回
    ----
    delta : (B, T)  扰动量（MW）
    """
    bus_num, time_num = input_load_i.shape

    # 识别 zero / nonzero bus（全时段均为零的节点）
    bus_is_zero = np.sum(np.abs(input_load_i), axis=1) < 1e-3
    zero_mask   = bus_is_zero.reshape(-1, 1)       # (B, 1)
    nonzero_mask = ~zero_mask                       # (B, 1)

    # zero bus: add renewable injection (negative load).
    renewable = perturb_zero * np.random.rand(bus_num, time_num) * zero_mask

    # nonzero bus: cap extra positive load per bus/time by perturb_nz.
    load_cap = perturb_nz * np.ones_like(input_load_i, dtype=float) * nonzero_mask
    compensation_ratio = max(0.0, float(load_compensation_ratio))
    renewable_sum = np.sum(renewable, axis=0, keepdims=True)

    # If positive-load capacity is insufficient, scale renewable down so the
    # generated sample does not collapse total net load.
    if compensation_ratio > 0:
        capacity_sum = np.sum(load_cap, axis=0, keepdims=True)
        valid_t = renewable_sum[0] > 1e-9
        scale = np.ones_like(renewable_sum)
        scale[:, valid_t] = np.minimum(
            1.0,
            capacity_sum[:, valid_t] / (renewable_sum[:, valid_t] * compensation_ratio),
        )
        renewable = renewable * scale
        renewable_sum = np.sum(renewable, axis=0, keepdims=True)

    delta_zero = -renewable

    # nonzero bus: add positive load and distribute it by the original load shape.
    extra_load_shape = (
        np.maximum(input_load_i, 0.0) + 0.05 * average_load
    ) * np.random.rand(bus_num, time_num) * nonzero_mask
    target_extra_load = renewable_sum * compensation_ratio
    delta_nz = _distribute_extra_load(extra_load_shape, target_extra_load, load_cap)

    return delta_zero + delta_nz


def _distribute_extra_load(shape, target, cap):
    """Distribute per-time positive load target by shape while respecting cap."""
    delta = np.zeros_like(shape, dtype=float)
    time_num = shape.shape[1]
    for t in range(time_num):
        remaining_target = float(target[0, t])
        if remaining_target <= 1e-9:
            continue

        residual_cap = cap[:, t].astype(float, copy=True)
        active = residual_cap > 1e-9
        while remaining_target > 1e-9 and np.any(active):
            weights = np.where(active, shape[:, t], 0.0)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                weights = active.astype(float)
                weight_sum = float(np.sum(weights))

            proposed = remaining_target * weights / weight_sum
            added = np.minimum(proposed, residual_cap)
            added_sum = float(np.sum(added))
            if added_sum <= 1e-9:
                break

            delta[:, t] += added
            residual_cap -= added
            remaining_target -= added_sum
            active = residual_cap > 1e-9

    return delta


def _count_switch_complexity(output):
    """Count generator on/off switching events in a (G, T) binary-like output."""
    if output.shape[1] <= 1:
        return 0
    return int(np.sum(np.abs(np.diff(output, axis=1)) > 0.5))


def _passes_switch_complexity_filter(new_output, base_output, min_ratio):
    base_complexity = _count_switch_complexity(base_output)
    new_complexity = _count_switch_complexity(new_output)
    required_complexity = int(np.floor(base_complexity * float(min_ratio)))
    return new_complexity >= required_complexity, new_complexity, base_complexity, required_complexity


# ── 主函数 ─────────────────────────────────────────────────────────────────


def generate_extra_data(
    new_data_num=1000,
    line_limit_rate=1,
    dict_name='training_data',
    storage_data_name='storage_data_1888rte.npz',
    case_dist='case_data',
    case_type='matpower',
    case_name='case1888rte_pro',
    ptdf_date='2017-01-01',
    new_add_num=10,
    perturb_zero_ratio=PERTURB_ZERO_RATIO,
    perturb_nz_ratio=PERTURB_NZ_RATIO,
    load_compensation_ratio=LOAD_COMPENSATION_RATIO,
    min_switch_complexity_ratio=MIN_SWITCH_COMPLEXITY_RATIO,
    save_interval=SAVE_INTERVAL,
    output_suffix='newinput',
):
    """
    生成新的训练数据。

    参数
    ----
    new_data_num          : int，生成的新样本数量（应 ≥ 原始样本总数，会覆盖整个数据集）
    line_limit_rate       : float，线路潮流约束倍率
    dict_name             : str，数据目录
    storage_data_name     : str，储能数据文件名
    case_dist/cut_type   : str，算例路径配置
    case_name             : str，算例名
    ptdf_date             : str，PTDF 日期
    new_add_num           : int，UC 求解时的候选机组数
    perturb_zero_ratio    : float，zero bus 扰动幅度（相对于 average_load）
    perturb_nz_ratio      : float，nonzero bus 每个时段新增正负荷上限
    min_switch_complexity_ratio : float，新输出切换次数相对 base 的最小保留比例
    save_interval         : int，每隔多少样本保存一次
    output_suffix         : str，输出文件后缀（如 'newinput'）
    """
    # ── 1. 加载原始数据 ───────────────────────────────────────────────
    base_data_file = f'all_training_data_{case_name}.npz'
    d = _load_array(os.path.join(dict_name, base_data_file))
    input_raw = d['input_data']           # (S, T, B)  归一化后
    average_load = float(np.asarray(d['average_load']).item())

    if input_raw.ndim == 3:
        input_raw = np.transpose(input_raw, (0, 2, 1))   # → (S, B, T)
    input_load = input_raw * average_load               # → MW 单位
    sample_num, bus_num, time_num = input_load.shape

    output_raw = d['output_data'].copy()
    if output_raw.ndim == 3:
        output_raw = np.transpose(output_raw, (0, 2, 1))
    base_output_ref = output_raw.copy()

    storage_data = _load_array(os.path.join(dict_name, storage_data_name))

    # ── 2. 加载数据模板 ───────────────────────────────────────────────
    ptdf_path = f'{case_dist}/{case_type}/{case_name}/{ptdf_date}.json.gz'
    with gzip.open(ptdf_path) as f:
        original_data = json.loads(f.read())

    key_list = list(original_data['Buses'].keys())

    unit_names, units = zip(*original_data['Generators'].items())
    NG = len(units)
    line_names, lines = (zip(*original_data['Transmission lines'].items())
                         if 'Transmission lines' in original_data else ([], []))
    NL = len(lines)

    # ── 3. 初始化输出数组（从 base 数据开始，逐步覆盖）──────────────
    out_data_name = f'all_training_data_{case_name}_{output_suffix}'
    out_npz = os.path.join(dict_name, out_data_name + '.npz')

    if os.path.exists(out_npz):
        # 已有数据：加载并从 base 重新初始化（避免历史数据干扰）
        h_in, h_out, h_en, h_l = get_history_data(dict_name, out_data_name + '.npz')
        history_input    = h_in.copy()
        history_output   = h_out.copy()
        history_enable   = h_en.copy()
        history_l        = h_l.copy()
    else:
        # 无历史：从 base 数据复制初始化
        history_input  = input_raw.copy()          # (S, B, T) 归一化
        history_output = output_raw.copy()
        history_enable = d['enable_data'].copy()
        history_l      = d['l_data'].copy()
        if history_l.ndim == 3:
            history_l = np.transpose(history_l, (0, 2, 1))

    # 扰动幅度（MW 单位）
    perturb_zero = perturb_zero_ratio * average_load
    perturb_nz    = perturb_nz_ratio    * average_load

    # ── 4. 加载约束预测模型（加速 UC 求解）──────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    enable_model_name = ('constraints_DNN_all_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002')
    enable_model, *_ = ml_load_model(enable_model_name, 'module_save')
    enable_model.eval()

    # ── 5. 主循环 ─────────────────────────────────────────────────────
    # 目标：在滤除无解/变化不足后，真正写入 new_data_num 个有效样本
    target_data_num = min(new_data_num, sample_num)
    max_attempts = max(sample_num * 10, target_data_num * 20)  # 安全上限，避免死循环

    skip_count = 0
    written_count = 0
    no_answer_count = 0
    attempt_count = 0

    while written_count < target_data_num and attempt_count < max_attempts:
        source_i = attempt_count % sample_num
        write_i = written_count
        attempt_count += 1
        print(f"[attempt {attempt_count}/{max_attempts}] 处理源样本 {source_i+1}/{sample_num} -> 写入位置 {write_i+1}/{target_data_num}（跳过 {skip_count}，无解 {no_answer_count}）")

        # ── 5a. 生成输入扰动 ────────────────────────────────────────
        delta = _gen_perturbation(
            input_load_i=input_load[source_i],     # (B, T) MW
            average_load=average_load,
            perturb_zero=perturb_zero,
            perturb_nz=perturb_nz,
            load_compensation_ratio=load_compensation_ratio,
        )

        # 扰动后的负荷（MW）
        input_i_mw = input_load[source_i] + delta    # (B, T) MW

        # 归一化后存入 history（用于后续模型输入）
        history_input[write_i] = input_i_mw / average_load

        # ── 5b. 更新数据模板中的 bus 负荷 ─────────────────────────────
        data_copy = json.loads(json.dumps(original_data))  # 深拷贝
        for k_i, k in enumerate(key_list):
            data_copy['Buses'][k]['Load (MW)'] = list(input_i_mw[k_i, :])

        # ── 5c. 预测约束使能（加速）──────────────────────────────────
        ml_input = (input_i_mw.T / average_load).astype(np.float32).flatten()
        ml_input_t = torch.from_numpy(ml_input).unsqueeze(0).to(device)
        with torch.no_grad():
            constraints_enable = enable_model(ml_input_t).squeeze(0).cpu().numpy()

        # ── 5d. 求解 UC ──────────────────────────────────────────────
        (model, result, u_start, y_start, z_start,
         p_start, r_start, c_start, l_start,
         enable_constraints, model_time, solve_t,
         load_data_t, round_num, no_answer) = \
            run_constraint_filter(
                data_copy,
                is_warm_start=False,
                use_new_input=True,
                use_storage=True,
                use_new_min_up_down_time=True,
                use_solar=False,
                add_more_constr=False,
                storage_data=storage_data,
                new_input=input_i_mw,
                line_limit_rate=line_limit_rate,
                new_add_num=new_add_num,
                add_constr_gate=0.7,
                initial_constraints=constraints_enable,
                use_initial_enable=True,
            )

        if no_answer:
            no_answer_count += 1
            print(f"  → 无可行解，跳过")
            continue

        # ── 5e. 提取结果 ─────────────────────────────────────────────
        new_output = np.array([
            [pyo.value(u_start[i, t]) for t in range(time_num)]
            for i in range(NG)
        ])  # (G, T)

        # ── 5f. 变化过滤 ────────────────────────────────────────────
        base_output = base_output_ref[source_i]              # (G, T)
        pass_complexity, new_complexity, base_complexity, required_complexity = \
            _passes_switch_complexity_filter(
                new_output, base_output, min_switch_complexity_ratio
            )

        if not pass_complexity:
            print(f"  -> switch complexity {new_complexity} < {required_complexity} "
                  f"(base={base_complexity}, ratio={min_switch_complexity_ratio}), skip")
            skip_count += 1
            continue

        # 写入
        history_output[write_i] = new_output
        history_enable[write_i, :] = enable_constraints
        history_l[write_i, :, :] = np.array([
            [pyo.value(l_start[i, t]) for t in range(time_num)]
            for i in range(NL)
        ])

        written_count += 1
        print(f"  -> switch complexity {new_complexity}/{base_complexity}, written={written_count}")

        # ── 5g. 定期保存 ────────────────────────────────────────────
        if written_count > 0 and written_count % save_interval == 0:
            _save(out_npz, history_input, history_output,
                  history_enable, history_l, average_load)
            print(f"  [自动保存 @ written={written_count}]")

    # ── 6. 最终保存 ──────────────────────────────────────────────────
    _save(out_npz, history_input, history_output,
          history_enable, history_l, average_load)

    print(f"\n完成：生成 {written_count}/{target_data_num} 个有效样本，"
          f"共尝试 {attempt_count} 次，跳过 {skip_count} 个（变化不足），无解 {no_answer_count} 个")
    return written_count


def _save(npz_path, input_data, output_data, enable_data, l_data, average_load):
    """保存为 .npz（统一 PyTorch (S,T,*) 惯例）。"""
    np.savez_compressed(
        npz_path,
        input_data  =np.transpose(input_data,  (0, 2, 1)),  # (S,T,B)
        output_data =np.transpose(output_data, (0, 2, 1)),   # (S,T,G)
        enable_data =enable_data,
        l_data      =np.transpose(l_data,      (0, 2, 1)),  # (S,T,L)
        average_load=float(average_load),
    )


# ── 命令行入口 ──────────────────────────────────────────────────────────────


if __name__ == '__main__':
    generate_extra_data(
        new_data_num     =1365,    # 覆盖全部样本，确保测试集也被重新求解
        line_limit_rate  =1,
        dict_name        ='training_data',
        storage_data_name='storage_data_1888rte.npz',
        case_dist       ='case_data',
        case_type       ='matpower',
        case_name       ='case1888rte_pro',
        ptdf_date       ='2017-01-01',
        new_add_num     =10,
        perturb_zero_ratio=2.0,
        perturb_nz_ratio  =1.5,
        load_compensation_ratio=1.0,
        min_switch_complexity_ratio=0.8,
        save_interval    =50,
        output_suffix    ='newinput',
    )
