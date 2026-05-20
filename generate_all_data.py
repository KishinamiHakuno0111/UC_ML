import os
import scipy.io
import gzip
import json
import pyomo.environ as pyo
import numpy as np
from uc_gurobi_constraint_filter import run_constraint_filter
import torch
from ml_basic_functions import  ml_load_model


def _load_array(path):
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    return scipy.io.loadmat(path)


def get_history_data(dict_name, data_name):
    d = _load_array(os.path.join(dict_name, data_name))
    history_input_data = d['input_data']
    history_output_data = d['output_data']
    history_output_enable_data = d['enable_data']
    history_l_data = d['l_data']
    # npz stores (S,T,B); callers expect (S,B,T)
    if history_input_data.ndim == 3:
        history_input_data = np.transpose(history_input_data, (0, 2, 1))
    if history_output_data.ndim == 3:
        history_output_data = np.transpose(history_output_data, (0, 2, 1))
    if history_l_data.ndim == 3:
        history_l_data = np.transpose(history_l_data, (0, 2, 1))
    return history_input_data, history_output_data, history_output_enable_data, history_l_data


def generate_extra_data(new_data_num=1000, save_gap=50, line_limit_rate=1, dict_name='training_data',training_data='training_data.mat',
                        storage_data_name = 'storage_data',
                        case_dist='case_data', case_type='matpower', case_name='case1888rte', ptdf_date='2017-01-01',
                        new_add_num=5):

    #  读取已有输入数据
    d = _load_array(os.path.join(dict_name, training_data))
    # .npz stores (S,T,B); callers expect (S,B,T)
    input_data_raw = d["input_data"]
    if input_data_raw.ndim == 3:
        input_data_raw = np.transpose(input_data_raw, (0, 2, 1))
    average_load = float(np.asarray(d["average_load"]).item())
    input_load = input_data_raw * average_load
    sample_num, bus_num, time_num = input_load.shape
    max_input_load = np.max(input_load, axis= 0)
    min_input_load = np.min(input_load, axis= 0) #  注意load必须为正才有意义
    delta_input_load = (max_input_load - min_input_load)

    # 读取储能数据
    storage_data = _load_array(os.path.join(dict_name, storage_data_name + '.mat'))

    #  生成随机矩阵
    random_matrix = 2*np.random.rand(bus_num, time_num, new_data_num)-1

    # #  生成随机矩阵
    # random_matrix = np.random.rand(sample_num, bus_num, time_num, new_data_num)
    #
    # #  开始归一化随机数矩阵，不同时间的不同节点，其sample_num个随机数加起来为1
    # rand_sum_matrix = np.sum(random_matrix, axis=0)
    # rand_sum_matrix_expanded = np.array([rand_sum_matrix for i in range(sample_num)]) #  扩展矩阵，方便后续点乘
    # normalized_random_matrix = random_matrix / rand_sum_matrix_expanded

    # 获得数据模板
    ptdf_path = case_dist+'/'+case_type+'/'+case_name+'/'+ptdf_date+'.json.gz'
    original_data_path = ptdf_path #  用于修改的数据模板（修改的是负载信息）
    file = gzip.open(original_data_path)
    original_data = json.loads(file.read())

    # 获得data中bus的索引
    key_list = list(original_data['Buses'].keys())
    key_num = len(key_list)

    #  获得基础数据（仅从 training_data 读取，不读旧的 all_training_data）
    history_data_name = 'all_' + training_data
    base_input_data, base_output_data, base_enable_data, base_l_data = get_history_data(dict_name, training_data)
    history_input_data = base_input_data
    history_output_data = base_output_data
    history_output_enable_data = base_enable_data
    history_l_data = base_l_data

    # 获取需要的数据
    unit_names, units = zip(*original_data['Generators'].items())
    NG = len(units)  # 获取NG
    line_names, lines = zip(*original_data['Transmission lines'].items()) if 'Transmission lines' in original_data else ([], [])
    NL = len(lines)

    #  开始循环
    input_save = []
    output_save = []
    output_enable_save = []
    l_save = []

    no_answer_num = 0

    # 设置约束预测模型
    # 获取gpu状态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    enable_model_name_list = ['constraints_DNN_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002']
    model_name = enable_model_name_list[0]
    model_dict_name = 'module_save'
    enable_model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(
        model_name, model_dict_name)

    for data_i in range(new_data_num):
        print(f"开始生成第{data_i}个数据：")

        # 利用随机数生成数据
        # 这是新的数据输入
        input_i = random_matrix[:, :, data_i] * delta_input_load + (min_input_load+max_input_load)/2

        b_num, t_num = input_i.shape
        ml_input_i = np.transpose(input_i)
        ml_input_i = ml_input_i.astype(np.float32)
        ml_input_i = np.reshape(ml_input_i, (b_num*t_num))
        ml_input_i = torch.from_numpy(ml_input_i)
        ml_input_i = ml_input_i.unsqueeze(0)
        ml_input_i = ml_input_i.to(device)
        enable_model = enable_model.to(device)

        forecast_enable = enable_model(ml_input_i)
        forecast_enable = forecast_enable.to(device_cpu)
        forecast_enable = forecast_enable.squeeze(0)
        forecast_enable = forecast_enable.detach().numpy()
        constraints_enable = forecast_enable
        use_initial_enable = True
        # constraints_enable = True

        #  计算结果

        model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, load_data_t, round_num, no_answer = \
            run_constraint_filter(original_data, is_warm_start=False, use_new_input=True, use_storage=True,
                                  use_new_min_up_down_time=True,
                                  use_solar=False, add_more_constr=False, storage_data=storage_data, new_input=input_i,
                                  line_limit_rate=line_limit_rate, new_add_num=new_add_num, add_constr_gate=0.7, initial_constraints=constraints_enable, use_initial_enable=use_initial_enable)
        if no_answer:
            no_answer_num = no_answer_num+1
            continue

        #储存结果
        input_array = input_i/average_load
        output_array = [[pyo.value(u_start[i, t]) for t in range(time_num)] for i in range(NG)]
        l_array = [[pyo.value(l_start[i, t]) for t in range(time_num)] for i in range(NL)] #  l没有归一化处理
        input_save.append(input_array)
        output_save.append(output_array)
        output_enable_save.append(enable_constraints)
        l_save.append(l_array)

        print(f"第{data_i}个数据生成完毕")

        if ((data_i % save_gap) == 0) & (data_i != 0):
            #  变为数组格式
            input_save = np.array(input_save)
            output_save = np.array(output_save)
            output_enable_save = np.array(output_enable_save)
            l_save = np.array(l_save)

            #  与旧数据整合在一起
            new_input = np.concatenate((history_input_data, input_save))
            new_output = np.concatenate((history_output_data, output_save))
            new_enable = np.concatenate((history_output_enable_data, output_enable_save))
            new_l = np.concatenate((history_l_data, l_save))

            print(f"现有数据个数为{len(new_input)}")

            file_path = os.path.join(dict_name, history_data_name)
            # 将数组保存为 .npz 文件
            np.savez_compressed(file_path,
                input_data=np.transpose(new_input, (0, 2, 1)),
                output_data=np.transpose(new_output, (0, 2, 1)),
                enable_data=new_enable,
                l_data=np.transpose(new_l, (0, 2, 1)),
                average_load=float(average_load))

            # 更新历史数据，并清空临时存储矩阵
            history_input_data = new_input
            history_output_data = new_output
            history_output_enable_data = new_enable
            history_l_data = new_l
            input_save = []
            output_save = []
            output_enable_save = []
            l_save = []

    #  变为数组格式
    input_save = np.array(input_save)
    output_save = np.array(output_save)
    output_enable_save = np.array(output_enable_save)
    l_save = np.array(l_save)

    #  与旧数据整合在一起
    new_input = np.concatenate((history_input_data, input_save))
    new_output = np.concatenate((history_output_data, output_save))
    new_enable = np.concatenate((history_output_enable_data, output_enable_save))
    new_l = np.concatenate((history_l_data, l_save))

    file_path = os.path.join(dict_name, history_data_name)
    # 将数组保存为 .npz 文件
    np.savez_compressed(file_path,
                        input_data=np.transpose(new_input, (0, 2, 1)),
                        output_data=np.transpose(new_output, (0, 2, 1)),
                        enable_data=new_enable,
                        l_data=np.transpose(new_l, (0, 2, 1)),
                        average_load=float(average_load))

    print(f"try to add {new_data_num} new data, {no_answer_num} data have no answer")
    return 0


if __name__ == '__main__':
    new_data_num = 1000
    save_gap = 50
    line_limit_rate = 1
    dict_name = 'training_data'
    training_data = 'training_data_case1888rte_pro.npz'
    storage_data_name = 'storage_data_case1888rte'
    case_dist = 'case_data'
    case_type = 'matpower'
    case_name = 'case1888rte'
    ptdf_date = '2017-01-01'
    new_add_num = 5

    generate_extra_data(new_data_num=new_data_num, save_gap=save_gap, line_limit_rate=line_limit_rate, dict_name=dict_name,
                        storage_data_name=storage_data_name, training_data = training_data,
                        case_dist=case_dist, case_type=case_type, case_name=case_name, ptdf_date=ptdf_date,
                        new_add_num=new_add_num)