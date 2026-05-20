"""
add_new_input.py — PTDF aggregation with per-channel normalization
=============================================================

公共函数
--------
aggregate_input(input_data, active_line_ptdf, output_data, l_data,
                average_load, enable_data,
                use_npz=True, use_normalization=True, save_path=None)
    将原始高维输入 (S,T,B) 按活跃线路 PTDF 投影聚合为低维 (S,T,NL+1)，
    对每个通道做 z-score 归一化，保存为 .npz（推荐）或 .mat。

命令行用法（保留原有行为）
--------------------------
    python add_new_input.py
        默认聚合 all_training_data_case1888rte_pro.mat，
        输出 all_training_data_case1888rte_pro_decreased_base.npz
"""

import os
import gzip
import json
import numpy as np
from generate_data import get_ptdf

# ── 内部辅助 ────────────────────────────────────────────────────────────────


def _load_data(path):
    """Load training data from .npz (preferred) or .mat, return dict."""
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io
    return scipy.io.loadmat(path)


def _detect_active_lines(enable_data, data_num, test_num):
    """
    从 enable_data 的训练部分推断活跃线路。
    enable_data : (S, L) array
    data_num    : 总样本数
    test_num    : 测试样本数
    返回        : active_lines (一维索引数组)
    """
    train_active = enable_data[:data_num - test_num, :]
    is_active = np.max(train_active, axis=0)
    return np.where(is_active > 0.5)[0]


def _build_active_ptdf_from_source(source, ptdf_data_path, data_num, test_num):
    """
    从 source 数据文件提取活跃线路，并构建对应的 active_line_ptdf 矩阵。

    参数
    ----
    source        : str 或 dict
                    - str: 路径，指向 .npz/.mat 训练数据文件
                    - dict: 已加载的数据字典（须含 'enable_data' 键）
    ptdf_data_path : str，PTDF JSON.gz 模板路径
    data_num       : int
    test_num       : int

    返回
    ----
    active_line_ptdf : (NB, NL) numpy数组，活跃线路的 PTDF 列向量拼接
    active_lines     : (NL,) 一维索引数组，活跃线路编号
    """
    if isinstance(source, str):
        d = _load_data(source)
        enable_data = d['enable_data']
    else:
        enable_data = source['enable_data']

    active_lines = _detect_active_lines(enable_data, data_num, test_num)

    with gzip.open(ptdf_data_path) as f:
        ptdf_template = json.loads(f.read())
    full_ptdf = get_ptdf(ptdf_template)          # (NL_all, NB)
    full_ptdf = np.array(full_ptdf)             # (NL_all, NB)

    active_line_ptdf = full_ptdf[active_lines, :]  # (NL, NB)
    active_line_ptdf = active_line_ptdf.T          # (NB, NL)

    return active_line_ptdf, active_lines


def _ptdf_projection(input_data, active_line_ptdf):
    """
    将原始输入按活跃线路 PTDF 矩阵投影。

    参数
    ----
    input_data        : (S, B, T)  numpy数组，原始归一化输入
    active_line_ptdf  : (B, NL)    PTDF 矩阵（仅活跃线路）

    返回
    ----
    agg : (S, NL+1, T)  聚合结果
          NL 个活跃通道 + 1 个残余通道（总负荷 - 各活跃线路投影和）
    """
    nb = active_line_ptdf.shape[0]
    nl = active_line_ptdf.shape[1]
    n_samples, _, n_time = input_data.shape

    # 合并 S 和 T 维度：(S,B,T) → (S*T, B)
    flat = input_data[:, :nb, :].reshape(n_samples * n_time, nb)     # (S*T, B)
    proj = flat @ active_line_ptdf                                        # (S*T, NL)
    total = flat.sum(axis=1, keepdims=True)                             # (S*T, 1)

    # 转回 (S, T, NL) 再转 (S, NL, T)
    proj_t = proj.reshape(n_samples, n_time, nl).transpose(0, 2, 1)     # (S, NL, T)
    residual = (total - proj.sum(axis=1, keepdims=True)).reshape(n_samples, 1, n_time)

    agg = np.concatenate([proj_t, residual], axis=1)                    # (S, NL+1, T)
    return agg


def _channel_stats(agg):
    """Return per-channel normalization statistics for (sample, channel, time)."""
    mean = agg.mean(axis=(0, 2))
    std = agg.std(axis=(0, 2)) + 1e-8
    return mean, std


def _normalize_channels(agg, stats=None):
    """
    对聚合结果的每个通道做 z-score 归一化（in-place）。
    修复 657x 的 scale 不平衡问题。
    """
    if stats is None:
        mean, std = _channel_stats(agg)
    else:
        mean, std = stats
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        std = np.asarray(std, dtype=np.float64).reshape(-1)
        if mean.shape[0] != agg.shape[1] or std.shape[0] != agg.shape[1]:
            raise ValueError(
                f"normalization stats length must be {agg.shape[1]}, "
                f"got mean={mean.shape[0]}, std={std.shape[0]}"
            )
        std = np.maximum(std, 1e-8)

    agg -= mean.reshape(1, -1, 1)
    agg /= std.reshape(1, -1, 1)
    return mean, std


def _to_sample_bus_time(input_data, bus_num):
    """Coerce input data to (sample, bus, time)."""
    input_data = np.asarray(input_data, dtype=np.float64)
    if input_data.ndim != 3:
        raise ValueError(f"input_data must be 3-D, got shape {input_data.shape}")
    if input_data.shape[1] == bus_num:
        return input_data
    if input_data.shape[2] == bus_num:
        return np.transpose(input_data, (0, 2, 1))
    raise ValueError(
        f"input_data shape {input_data.shape} does not match bus_num={bus_num}; "
        "expected (S,B,T) or (S,T,B)."
    )


def _normalization_stats_from_source(source, active_line_ptdf, output_key='input_data'):
    """Build aggregation normalization stats from a reference dataset."""
    d = _load_data(source)
    if 'aggregation_mean' in d and 'aggregation_std' in d:
        mean = np.asarray(d['aggregation_mean']).reshape(-1)
        std = np.asarray(d['aggregation_std']).reshape(-1)
        if mean.shape[0] == active_line_ptdf.shape[1] + 1:
            return mean, std

    source_input = _to_sample_bus_time(d[output_key], active_line_ptdf.shape[0])
    source_agg = _ptdf_projection(source_input, active_line_ptdf)
    return _channel_stats(source_agg)


def _save_aggregated(agg, output_data, l_data, average_load, enable_data,
                     use_npz, save_path, active_lines=None,
                     normalization_mean=None, normalization_std=None,
                     normalization_source=None):
    """将聚合结果保存为 .npz 或 .mat。"""
    # agg 当前为 (S, NL+1, T)，转为 (S, T, NL+1) 匹配 PyTorch 惯例
    agg_t = np.transpose(agg, (0, 2, 1))

    if use_npz:
        payload = dict(
            input_data=agg_t,
            output_data=output_data,
            l_data=l_data,
            average_load=float(average_load),
            enable_data=enable_data,
        )
        if active_lines is not None:
            payload['active_lines'] = np.asarray(active_lines, dtype=np.int64)
        if normalization_mean is not None and normalization_std is not None:
            payload['aggregation_mean'] = np.asarray(normalization_mean, dtype=np.float64)
            payload['aggregation_std'] = np.asarray(normalization_std, dtype=np.float64)
        if normalization_source is not None:
            payload['normalization_source'] = np.asarray(str(normalization_source))
        np.savez_compressed(save_path, **payload)
    else:
        import scipy.io
        payload = {'input_data': agg, 'output_data': output_data,
                   'l_data': l_data, 'average_load': average_load,
                   'enable_data': enable_data}
        if active_lines is not None:
            payload['active_lines'] = np.asarray(active_lines, dtype=np.int64)
        if normalization_mean is not None and normalization_std is not None:
            payload['aggregation_mean'] = np.asarray(normalization_mean, dtype=np.float64)
            payload['aggregation_std'] = np.asarray(normalization_std, dtype=np.float64)
        if normalization_source is not None:
            payload['normalization_source'] = str(normalization_source)
        scipy.io.savemat(save_path, payload)


# ── 公共 API ────────────────────────────────────────────────────────────────


def aggregate_input(
    input_data,
    active_line_ptdf,
    output_data,
    l_data,
    average_load,
    enable_data,
    use_npz=True,
    use_normalization=True,
    normalization_stats=None,
    normalization_source=None,
    active_lines=None,
    save_path=None,
):
    """
    将原始高维输入按活跃线路 PTDF 投影聚合，并可选保存。

    参数
    ----
    input_data        : (S, T, B) 或 (S, B, T)  numpy数组
                        若为 (S, T, B) 会自动转为 (S, B, T)。
    active_line_ptdf  : (B, NL)  PTDF 矩阵（仅活跃线路）。
                        形状应为 (nb_buses, nl_active_lines)。
    output_data       : (S, G, T) 或 (S, T, G)  numpy数组，原样透传。
    l_data            : (S, L, T) 或 (S, T, L)  numpy数组，原样透传。
    average_load      : float，平均负荷标量。
    enable_data       : (S, L)  约束使能数组，原样透传。
    use_npz           : bool，True → 保存 .npz（推荐）；False → 保存 .mat。
    use_normalization : bool，True → 对每个聚合通道做 z-score 归一化（推荐，
                        修复 657x scale 不平衡）。设为 False 可保留原始量纲。
    normalization_stats : optional (mean, std) tuple used for z-score normalization.
    normalization_source: optional reference dataset path recorded in the output metadata.
    active_lines      : optional active line indices recorded in the output metadata.
    save_path         : str 或 None，若非 None 则保存到该路径。
                        .npz 时自动按 PyTorch (S,T,B) 惯例存储。
                        若为 None 则不保存，仅返回聚合结果。

    返回
    ----
    agg : (S, T, NL+1)  numpy数组，聚合后（且归一化后）的输入。
          保存时同样存为 (S, T, NL+1)。
    """
    # ── 1. 统一为 (S, B, T) ────────────────────────────────────────────────
    input_data = _to_sample_bus_time(input_data, active_line_ptdf.shape[0])

    # ── 2. PTDF 投影 ──────────────────────────────────────────────────────
    agg = _ptdf_projection(input_data, active_line_ptdf)

    # ── 3. 每通道 z-score 归一化（在 transpose 之前，agg 为 (S,NL+1,T)）────
    if use_normalization:
        std0 = agg[:, 0, :].std()
        stdN = agg[:, -1, :].std()
        normalization_mean, normalization_std = _normalize_channels(agg, normalization_stats)
        std0_after = agg[:, 0, :].std()
        stdN_after = agg[:, -1, :].std()
        print(f"  Normalization: ch0 std {std0:.4f}→{std0_after:.4f},  "
              f"ch{agg.shape[1]-1} std {stdN:.4f}→{stdN_after:.4f}")
    else:
        normalization_mean = None
        normalization_std = None

    # ── 4. 保存 ────────────────────────────────────────────────────────────
    if save_path is not None:
        _save_aggregated(agg, output_data, l_data, average_load,
                         enable_data, use_npz, save_path,
                         active_lines=active_lines,
                         normalization_mean=normalization_mean,
                         normalization_std=normalization_std,
                         normalization_source=normalization_source)
        print(f"  Saved: {save_path}")

    # 返回 (S, T, NL+1)
    return np.transpose(agg, (0, 2, 1))


def aggregate_from_file(
    data_path,
    ptdf_data_path,
    active_line_ptdf=None,
    active_constraints_source=None,
    data_num=None,
    test_num=100,
    use_npz=True,
    use_normalization=True,
    normalization_source=None,
    save_path=None,
    output_key='input_data',
):
    """
    从数据文件直接完成聚合（含 PTDF 投影 + 归一化 + 保存）。

    参数
    ----
    data_path                  : str，训练数据文件路径。
                                优先加载同名 .npz，不存在则回退 .mat。
    ptdf_data_path            : str，PTDF 模板 JSON.gz 路径
                                （从中提取完整 PTDF 矩阵）。
    active_line_ptdf          : (B, NL) numpy数组 或 None。
                                手动指定 PTDF 投影矩阵。
    active_constraints_source  : str, dict, None
                                指定用于确定"活跃线路"的参考数据来源，
                                确保聚合输出的通道数与指定模型一致。
                                - str: 训练数据文件路径，从其 enable_data 推断活跃线路
                                       （例如传入 'training_data/all_training_data_case1888rte_pro.npz'
                                        以复用基础训练的活跃线路集）
                                - dict: 已加载的数据字典（含 'enable_data'）
                                - None: 从 data_path 自身推断活跃线路
    data_num                  : int 或 None，默认 None（取 data_path 全部样本数）。
                                用于 active_line 检测的分割点。
    test_num                  : int，默认 100。用于 active_line 检测。
    use_npz                   : bool，默认 True。
    use_normalization         : bool，默认 True（推荐，修复 657x scale 不平衡）。
    normalization_source      : str 或 None。若非 None，则用该参考数据的聚合均值/标准差
                                做 z-score，适用于无重训练迁移测试。
    save_path                 : str 或 None。若为 None，自动生成 *_decreased_base.npz。
    output_key                : str，默认 'input_data'。

    返回
    ----
    agg : (S, T, NL+1) numpy数组
    """
    # ── 1. 加载训练数据 ────────────────────────────────────────────────────
    d = _load_data(data_path)
    input_data = d[output_key]
    n_samples = input_data.shape[0]

    if data_num is None:
        data_num = n_samples

    average_load = float(np.asarray(d['average_load']).item())
    output_data = d['output_data']
    l_data = d['l_data']
    enable_data = d['enable_data']

    # ── 2. 活跃线路 + PTDF 矩阵 ────────────────────────────────────────────
    if active_line_ptdf is None:
        if active_constraints_source is not None:
            # 从外部参考数据构建 PTDF（确保与指定模型输入尺寸一致）
            active_line_ptdf, active_lines = _build_active_ptdf_from_source(
                active_constraints_source, ptdf_data_path, data_num, test_num)
            print(f"  Using external active constraints from source: "
                  f"{len(active_lines)} lines = {active_lines.tolist()}")
        else:
            # 从 data_path 自身推断
            active_lines = _detect_active_lines(enable_data, data_num, test_num)
            with gzip.open(ptdf_data_path) as f:
                ptdf_template = json.loads(f.read())
            full_ptdf = np.array(get_ptdf(ptdf_template))  # (NL_all, NB)
            active_line_ptdf = full_ptdf[active_lines, :].T    # (NB, NL)
            print(f"  Auto-detected {len(active_lines)} active lines: "
                  f"{active_lines.tolist()}")
    else:
        active_line_ptdf = np.asarray(active_line_ptdf)
        active_lines = None

    normalization_stats = None
    if use_normalization and normalization_source is not None:
        normalization_stats = _normalization_stats_from_source(
            normalization_source, active_line_ptdf, output_key=output_key)
        print(f"  Using normalization stats from source: {normalization_source}")

    # ── 3. 生成保存路径 ───────────────────────────────────────────────────
    if save_path is None:
        stem = os.path.splitext(data_path)[0]
        ext = '.npz' if use_npz else '.mat'
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        if isinstance(active_constraints_source, str):
            src_name = os.path.splitext(os.path.basename(active_constraints_source))[0]
            # 去掉 "all_training_data_" 前缀，保留核心标识
            src_tag = src_name.replace('all_training_data_', '')
            save_path = stem + '_decreased_base_' + src_tag + ext
        else:
            save_path = stem + '_decreased_base' + ext

    # ── 4. 聚合 ────────────────────────────────────────────────────────────
    print(f"Aggregating {data_path}")
    print(f"  input shape: {input_data.shape}  →  output: (S,T,{active_line_ptdf.shape[1]+1})")
    agg = aggregate_input(
        input_data=input_data,
        active_line_ptdf=active_line_ptdf,
        output_data=output_data,
        l_data=l_data,
        average_load=average_load,
        enable_data=enable_data,
        use_npz=use_npz,
        use_normalization=use_normalization,
        normalization_stats=normalization_stats,
        normalization_source=normalization_source,
        active_lines=active_lines,
        save_path=save_path,
    )
    print(f"  Done. output shape: {agg.shape}")
    return agg


# ── 命令行入口 ──────────────────────────────────────────────────────────────


if __name__ == '__main__':
    # 演示：用 base 数据的活跃约束对 newinput 数据做聚合
    # 这样聚合输出的通道数与 units_transformer_..._decreased_base 模型一致
    dict_name = 'training_data'
    newinput_path = os.path.join(dict_name, 'all_training_data_case1888rte_pro_newinput.npz')

    base_constraints = os.path.join(dict_name, 'all_training_data_case1888rte_pro.npz')

    ptdf_data_path = 'case_data/matpower/case1888rte_pro/2017-01-01.json.gz'

    aggregate_from_file(
        data_path=newinput_path,
        ptdf_data_path=ptdf_data_path,
        active_constraints_source=base_constraints,
        data_num=1100,
        test_num=100,
        use_npz=True,
        use_normalization=True,
        normalization_source=base_constraints,
    )
