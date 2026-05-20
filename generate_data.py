import os
import gzip
import json
import numpy as np
import time
import scipy.io
from uc_gurobi_constraint_filter import run_constraint_filter
from ml_basic_functions import magnify_peak_valley_difference, get_load


USE_NPZ = True  # True → save .npz (PyTorch axis order: S,T,B / S,T,G)


def try_get_from_list(x, t):
    return x[t] if type(x) == list else x


def get_ptdf(data):
    bus_names, buses = zip(*data['Buses'].items())
    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
    bus_idx_map = {bus_name: idx for idx, bus_name in enumerate(bus_names)}
    # compute ptdf
    susceptance = np.diag([line['Susceptance (S)'] for line in lines])
    incidence = np.zeros([len(lines), len(buses)])
    for line_idx, line in enumerate(lines):
        source_bus_idx = bus_idx_map[line['Source bus']]
        target_bus_idx = bus_idx_map[line['Target bus']]
        incidence[line_idx, source_bus_idx] = 1
        incidence[line_idx, target_bus_idx] = -1
    incidence = incidence[:, 1:]
    laplacian = incidence.T @ susceptance @ incidence
    ptdf = np.zeros([len(lines), len(buses)])
    # ptdf[:, 1:] = susceptance @ incidence @ np.linalg.inv(laplacian)
    ptdf[:, 1:] = np.linalg.solve(laplacian, incidence.T @ susceptance).T
    return ptdf


def get_class(ptdf, active_lines, gate):
    active_line_num = len(active_lines)
    bus_num = np.size(ptdf, 1)
    is_affected = np.zeros([bus_num, active_line_num]) #  (i, j), 1代表第i个机组对第j条（起作用的）线路的ptdf大于0.005
    is_affected = is_affected.tolist()
    for i in range(bus_num):
        for j in range(active_line_num):
            line_index = active_lines[j]
            if abs(ptdf[line_index, i]) > gate:
                is_affected[i][j] = 1
    return is_affected


def generate_365_data(case_dict='case_data', case_type='matpower', case_name='case1888rte', ptdf_date='2017-01-01', data_year='2017',
                      data_dist_name = 'training_data', storage_data_name = 'storage_data', new_add_num=5, magnify_rate = 1.05):
    start_time = time.time()
    ptdf_path = case_dict+'/'+case_type+'/'+case_name+'/'+ptdf_date+'.json.gz'
    file = gzip.open(ptdf_path)
    data = json.loads(file.read())
    ptdf = get_ptdf(data)

    # bus_list = data['Buses'].keys()
    # from_bus = {}
    # to_bus = {}
    # bus_line_dict = {}
    #
    # for k in bus_list:
    #     from_bus[k] = 0
    #     to_bus[k] = 0
    #     bus_line_dict[k] = []
    #
    # line_k = data['Transmission lines'].keys()
    # for l in line_k:
    #     line = data['Transmission lines'][l]
    #     source_bus = line['Source bus']
    #     trans_bus = line['Target bus']
    #     from_bus[source_bus] = from_bus[source_bus] + 1
    #     to_bus[trans_bus] = to_bus[trans_bus] + 1
    #     bus_line_dict[source_bus].append(l)
    #     bus_line_dict[trans_bus].append(l)
    #
    # for k in bus_list:
    #     if (from_bus[k] + to_bus[k]) == 1:
    #         print(k)
    #         l = bus_line_dict[k][0]
    #         print(bus_line_dict[k][0])
    #         line = data['Transmission lines'][l]
    #         print(line)

    active_lines = []
    line_limit_rate = 1
    is_decreased = False

    gate = 0.005

    ## 测试代码
    # data_num = 365
    # input_data = np.ones([data_num, NB, NT])
    # output_data = np.ones([data_num, NG, NT])
    # #  将input_data 和 output_data变为array
    # input_array = np.array([[[input_data[i, j, t] for t in range(NT)] for j in range(NB)] for i in range(data_num)]) #  input_array[i, j, t]表示第i个数据，第k个节点，第t时间
    # average_load = sum(sum(sum(input_array)))/(data_num*NT*NB)
    # input_array = input_array/average_load #  进行归一化
    # output_array = np.array([[[output_data[i, j, t] for t in range(NT)] for j in range(NG)] for i in range(data_num)])
    #
    # if is_decreased:
    #     input_array_decreased = np.zeros([data_num, class_len, NT])
    #     for i in range(NB):
    #         class_i = int(bus_classes[i])
    #         input_array_decreased[:, class_i, :] = input_array_decreased[:, class_i, :] + input_array[:, i, :]
    #     #  存储文件
    #     file_name = 'training_data_decreased.mat'
    #     file_path = os.path.join(file_name)
    #     # 将数组保存为 .mat 文件
    #     scipy.io.savemat(file_path, {'input_data': input_array_decreased, 'output_data': output_array, 'average_load': average_load})
    # else:
    #     #  存储文件
    #     file_name = 'training_data.mat'
    #     file_path = os.path.join(file_name)
    #
    #     # 将数组保存为 .mat 文件
    #     scipy.io.savemat(file_path, {'input_data': input_array, 'output_data': output_array, 'average_load': average_load})

    # 如果要使用，注意需要修改多个地方的算例名称，最好先整理成函数形式

    path = case_dict+'/'
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    data_index = 0
    input_data = {}  # 输出为u(i, j, t),其中i为数据索引,j为节点编号，t为时间,未进行归一化
    output_data = {}  # 输出为u(i, j, t),其中i为数据索引,j为机组编号，t为时间
    output_data_l_enable = {}
    l_data = {}
    for month in range(1, 13):  # 循环获得日期
        if month < 10:
            month_str = '0' + str(month)
        else:
            month_str = str(month)
        day_num = month_day[month - 1]
        for day in range(1, day_num + 1):
            if day < 10:
                day_str = '0' + str(day)
            else:
                day_str = str(day)
            file_name = data_year+'-' + month_str + '-' + day_str + '.json.gz'
            case = case_type+'/'+case_name+'/'
            file = gzip.open(path + case + file_name)
            data = json.loads(file.read())

            # 获取新的负载
            load_data = get_load(data)
            new_load_data = magnify_peak_valley_difference(load_data, magnify_rate)

            # 获取需要的数据
            unit_names, units = zip(*data['Generators'].items())
            NG = len(units)  # 获取NG
            line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
            NL = len(lines)
            normal_flow_limit = [line_limit_rate * line.get('Normal flow limit (MW)', 1e6) for line in
                                 lines]  # 获取潮流约束上限

            time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
            time_step = data['Parameters'].get('Time step (min)', 60)
            assert (60 % time_step == 0)
            time_multiplier = int(60 / time_step)
            NT = time_horizon * time_multiplier  # 获取NT

            bus_names, buses = zip(*data['Buses'].items())
            NB = len(buses)  # 获取NB

            # 在这里改变输入,使用新输入
            input_save_dict = {(data_index, i, t): new_load_data[i][t] for i in range(NB) for t in range(NT)}
            input_data.update(input_save_dict)

            # 获取储能信息
            storage_data_path = os.path.join(data_dist_name, storage_data_name + '.npz')
            if not os.path.exists(storage_data_path):
                storage_data_path = os.path.join(data_dist_name, storage_data_name + '.mat')
            storage_data = dict(np.load(storage_data_path, allow_pickle=True)) \
                if storage_data_path.endswith('.npz') \
                else scipy.io.loadmat(storage_data_path)

            model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, load_data_t, round_num, no_answer = \
                run_constraint_filter(data, is_warm_start=False, use_new_input=True, use_storage=True,
                                      use_new_min_up_down_time=True,
                                      use_solar=False, add_more_constr=False, storage_data=storage_data,
                                      new_input=new_load_data,
                                      line_limit_rate=line_limit_rate, new_add_num=new_add_num, add_constr_gate=0.7)
            if no_answer:
                continue
            if is_decreased:
                #  获取紧凑约束
                l_array = [[abs(l_start[i, t]) for i in range(NL)] for t in range(NT)]
                l_array_max = max(l_array)
                l_array_max = np.array(l_array_max)
                normal_flow_limit = np.array(normal_flow_limit)
                delta = normal_flow_limit - l_array_max
                is_active = (delta < 1e-3)
                for i in range(NL):
                    if is_active[i]:
                        active_line = i
                        if not (active_line in active_lines):
                            active_lines.append(active_line)

            output_save_dict = {(data_index, i, t): u_start[i, t] for i in range(NG) for t in range(NT)}
            output_save_dict_l_enable = {(data_index, i): enable_constraints[i] for i in range(NL)}
            l_save_dict = {(data_index, i, t): l_start[i, t] for i in range(NL) for t in range(NT)}
            output_data.update(output_save_dict)
            l_data.update(l_save_dict)
            output_data_l_enable.update(output_save_dict_l_enable)
            print(f'第{data_index}个数据计算完成，日期为{file_name}')
            data_index = data_index + 1

    if is_decreased:
        gate = 0.005
        bus_line_active = get_class(ptdf, active_lines, gate)  # bus和line之间的起作用关系
        bus_classes = np.zeros(NB)  # 根据其作用约束来分组
        affected_list = []
        class_num = 0
        for i in range(NB):
            line_is_affected = bus_line_active[i]
            if not (line_is_affected in affected_list):
                affected_list.append(line_is_affected)
                bus_classes[i] = class_num
                class_num = class_num + 1
            else:
                class_index = affected_list.index(line_is_affected)
                bus_classes[i] = class_index
        class_len = len(affected_list)

    data_num = data_index
    #  将input_data 和 output_data变为array
    input_array = np.array([[[input_data[i, j, t] for t in range(NT)] for j in range(NB)] for i in
                            range(data_num)])  # input_array[i, j, t]表示第i个数据，第k个节点，第t时间
    average_load = sum(sum(sum(input_array))) / (data_num * NT * NB)
    input_array = input_array / average_load  # 进行归一化
    output_array = np.array([[[output_data[i, j, t] for t in range(NT)] for j in range(NG)] for i in range(data_num)])
    output_array_l_enable = np.array([[output_data_l_enable[i, j] for j in range(NL)] for i in range(data_num)])
    l_array = np.array([[[l_data[i, j, t] for t in range(NT)] for j in range(NL)] for i in range(data_num)])

    if is_decreased:
        input_array_decreased = np.zeros([data_num, class_len, NT])
        for i in range(NB):
            class_i = int(bus_classes[i])
            input_array_decreased[:, class_i, :] = input_array_decreased[:, class_i, :] + input_array[:, i, :]
        #  存储文件
        dict_name = data_dist_name
        file_name = 'training_data_'+case_name + '_' + storage_data_name +'_decreased.mat'
        npz_name  = 'training_data_'+case_name + '_' + storage_data_name +'_decreased.npz'
        file_path = os.path.join(dict_name, file_name)
        npz_path  = os.path.join(dict_name, npz_name)
        if USE_NPZ:
            # (S,B,T)→(S,T,B)
            np.savez_compressed(npz_path,
                input_data=np.transpose(input_array_decreased, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                average_load=float(average_load),
                active_lines=np.array(active_lines))
        else:
            import scipy.io
            scipy.io.savemat(file_path,
                             {'input_data': input_array_decreased, 'output_data': output_array, 'l_data': l_array,
                              'average_load': average_load, 'active_lines': active_lines})
    else:
        #  存储文件
        dict_name = data_dist_name
        file_name = 'training_data_'+case_name + '_' + storage_data_name +'.mat'
        file_name_all_data = 'all_training_data_'+case_name + '_' + storage_data_name +'.mat'
        npz_name  = 'training_data_'+case_name + '_' + storage_data_name +'.npz'
        npz_name_all = 'all_training_data_'+case_name + '_' + storage_data_name +'.npz'
        file_path = os.path.join(dict_name, file_name)
        file_path_all_data = os.path.join(dict_name, file_name_all_data)
        npz_path  = os.path.join(dict_name, npz_name)
        npz_path_all = os.path.join(dict_name, npz_name_all)

        if USE_NPZ:
            common = dict(
                average_load=float(average_load),
                active_lines=np.array(active_lines),
                enable_data=output_array_l_enable,
            )
            np.savez_compressed(npz_path,
                input_data=np.transpose(input_array, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                **common)
            np.savez_compressed(npz_path_all,
                input_data=np.transpose(input_array, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                **common)
        else:
            import scipy.io
            scipy.io.savemat(file_path, {'input_data': input_array, 'output_data': output_array, 'l_data': l_array,
                                         'average_load': average_load, 'active_lines': active_lines,
                                         'enable_data': output_array_l_enable})
            scipy.io.savemat(file_path_all_data, {'input_data': input_array, 'output_data': output_array, 'l_data': l_array,
                                                  'average_load': average_load, 'active_lines': active_lines,
                                                  'enable_data': output_array_l_enable})
        print(f"generate {data_num} data over!")
    end_time = time.time()
    all_time = end_time-start_time
    print(f"generate {data_num} data cost {all_time} seconds! ")
    return 0

if __name__ == '__main__':
    case_dict = 'case_data'
    case_type = 'matpower'
    case_name = 'case1888rte'
    ptdf_date = '2017-01-01'
    data_year = '2017'
    data_dist_name = 'training_data'
    storage_data_name = 'storage_data_case1888rte_0%'
    new_add_num = 5
    magnify_rate = 1
    generate_365_data(case_dict=case_dict, case_type=case_type, case_name=case_name, ptdf_date=ptdf_date,
                      data_year=data_year, data_dist_name=data_dist_name, storage_data_name=storage_data_name,
                      new_add_num=new_add_num, magnify_rate=magnify_rate)
# 需要更新generate_data, generate_all_data, run_uc, constraints_filter, storage_data 5个文件到服务器
# 注意测试时需要更换storage_data
# 1.生成1000个数据后对比出效果（用原来的sto_data和1.05倍率）
# 2.使用新的stodata以及2倍倍率，出效果
# 3.使用新的stodata以及新能源，出效果
# ppt先介绍问题的复杂性，时空问题，新能源储能，然后由此带来的DNN挑战，所以采用新的思路

#case30的负载大概为300,设定30%的储能容量即为90