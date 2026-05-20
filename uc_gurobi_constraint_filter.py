from unit_commitment_gurobi import run_uc, map_list, get_load, get_solar_gen, get_data, magnify_peak_valley_difference, ml_load_model, timeseries
import time
import gzip
import json
import numpy as np
import os


def _load_storage_or_solar(npz_path):
    """Load storage/solar data: try .npz first (new), then .mat (legacy)."""
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    mat_path = npz_path.replace('.npz', '.mat')
    if os.path.exists(mat_path):
        import scipy.io
        return scipy.io.loadmat(mat_path)
    raise FileNotFoundError(f"Neither {npz_path} nor {mat_path} found")


# 1.把Input格式再修改 2.计算大规模算例 3.使用矩阵来算潮流 4.使用LSTM与transformer预测起作用约束
def get_overload_lines(l_start, normal_flow_limit, add_num, NT, NL):
    l_array = np.array([[l_start[i, t] for t in range(NT)] for i in range(NL)])
    l_array_abs = abs(l_array)
    limit_matrix = np.matmul(np.reshape(normal_flow_limit, [NL, 1]), np.ones([1, NT]))
    overload_percentage = l_array_abs/limit_matrix

    for i in range(NL):
        # 排除未约束线路的影响
        if normal_flow_limit[i] >= 1e6:
            overload_percentage[i, :] = 0

    is_overload = True if np.max(overload_percentage, axis=(0, 1)) > 1 else False

    max_overload_percentage = np.max(overload_percentage, axis=1)

    sorted_max_overload_percentage = np.sort(max_overload_percentage)
    sorted_max_overload_percentage = sorted_max_overload_percentage[::-1]
    largest_values = sorted_max_overload_percentage[0:add_num]

    for i in range(add_num-1, -1, -1):
        if largest_values[i] < 0.7:
            largest_values = np.delete(largest_values, i)

    largest_indices = [np.where(max_overload_percentage == value) for value in largest_values]
    return max_overload_percentage, largest_indices, is_overload, largest_values

# 考虑使用矩阵运算来加速
def caculate_line_flow_save(data, p_start, use_storage = False, use_new_input=False, storage_data = 0, new_input=0, pch_start=0, pdis_start=0, line_limit_rate=1):

    t_start = time.time()
    unit_names, units = zip(*data['Generators'].items())
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
    # ptdf[abs(ptdf) < 0.005] = 0

    time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
    time_step = data['Parameters'].get('Time step (min)', 60)
    assert(60 % time_step == 0)
    time_multiplier = int(60 / time_step)

    NT = time_horizon * time_multiplier
    NG = len(units)
    NB = len(buses)
    NL = len(lines)
    T_SET = range(NT)  # index set of time periods, start from 0
    G_SET = range(NG)
    B_SET = range(NB)
    L_SET = range(NL)

    B_TO_L_SET = [np.where(abs(ptdf[i, :]) > 0)[0].tolist() for i in L_SET]
    G_TO_L_SET = [[unit_idx for unit_idx in G_SET if bus_idx_map[units[unit_idx]['Bus']] in B_TO_L_SET[i]] for i in L_SET] # 有点慢

    bus_to_line_factor = [{j: ptdf[i, j] for j in B_TO_L_SET[i]} for i in L_SET] # 有点点慢
    unit_to_line_factor = [{j: ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_TO_L_SET[i]} for i in L_SET]

    normal_flow_limit = np.array(map_list(lines, lambda x: x.get('Normal flow limit (MW)', 1e6))) * line_limit_rate

    if use_new_input:
        load_demand = new_input
    else:
        load_demand = np.array(map_list(buses, lambda x: timeseries(x['Load (MW)'], NT)))

    if use_storage:
        # 录入储能设施数据
        sto_initial_power = storage_data['initial_power']
        sto_storage_position = storage_data['storage_position']
        sto_num = len(sto_initial_power)

    if use_storage:
        l_value = {(i, t): sum(unit_to_line_factor[i][j] * p_start[j, t] for j in G_TO_L_SET[i]) - sum(bus_to_line_factor[i][j]*load_demand[j, t] for j in B_TO_L_SET[i])
               +sum(ptdf[i][int(sto_storage_position[j])]*(pdis_start[j, t]-pch_start[j, t]) for j in range(sto_num)) for i in L_SET for t in T_SET} # 非常慢
    else:
        l_value = {(i, t): sum(unit_to_line_factor[i][j] * p_start[j, t] for j in G_TO_L_SET[i]) - sum(
            bus_to_line_factor[i][j] * load_demand[j, t] for j in B_TO_L_SET[i]) for i in L_SET for t in T_SET}

    t_end = time.time()
    print(f"计算潮流花费时间为{t_end-t_start}")

    return l_value, NT, NL, normal_flow_limit


def caculate_line_flow(data, p_start, use_storage = False, use_new_input=False, storage_data = 0, new_input=0, pch_start=0, pdis_start=0, line_limit_rate=1):
    t_start = time.time()
    unit_names, units = zip(*data['Generators'].items())
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
    # ptdf[abs(ptdf) < 0.005] = 0

    time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
    time_step = data['Parameters'].get('Time step (min)', 60)
    assert(60 % time_step == 0)
    time_multiplier = int(60 / time_step)

    NT = time_horizon * time_multiplier
    NG = len(units)
    NB = len(buses)
    NL = len(lines)
    T_SET = range(NT)  # index set of time periods, start from 0
    G_SET = range(NG)
    B_SET = range(NB)
    L_SET = range(NL)

    normal_flow_limit = np.array(map_list(lines, lambda x: x.get('Normal flow limit (MW)', 1e6))) * line_limit_rate
    if use_new_input:
        load_demand = new_input
    else:
        load_demand = np.array(map_list(buses, lambda x: timeseries(x['Load (MW)'], NT)))

    if use_storage:
        # 录入储能设施数据
        if len(storage_data['storage_position']) != 1:
            sto_storage_position = storage_data['storage_position']
        else:
            sto_storage_position = storage_data['storage_position'][0]
        sto_num = len(sto_storage_position)
        sto_to_line_matrix = np.array([[ptdf[i, sto_storage_position[j]] for j in range(sto_num)] for i in L_SET])

    unit_to_line_matrix = np.array([[ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_SET] for i in L_SET])
    p_matrix = np.array([[p_start[i, t] for t in T_SET] for i in G_SET])
    pdis_ch_matrix = np.array([[pdis_start[i, t]-pch_start[i, t] for t in T_SET] for i in range(sto_num)])
    if use_storage:
        l_value = np.matmul(unit_to_line_matrix, p_matrix) - np.matmul(ptdf, load_demand) + np.matmul(sto_to_line_matrix, pdis_ch_matrix)
    else:
        l_value = np.matmul(unit_to_line_matrix, p_matrix) - np.matmul(ptdf, load_demand)

    t_end = time.time()
    print(f"计算潮流花费时间为{t_end-t_start}")
    return l_value, NT, NL, normal_flow_limit


def run_constraint_filter(data, is_warm_start = False, use_new_input = False, use_storage = False,use_initial_enable=False,add_more_constr=False, use_new_min_up_down_time = False,
           use_solar = False ,u_start = 0, y_start = 0, z_start = 0, new_input = 0, storage_data = 0, line_limit_rate = 1, solar_gen = 0, new_add_num = 5, initial_constraints=0, add_constr_gate=0.7, fix_int = False, time_limit = 20, mip_gap = 1e-3):

    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
    time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
    time_step = data['Parameters'].get('Time step (min)', 60)
    assert(60 % time_step == 0)
    time_multiplier = int(60 / time_step)
    NL = len(lines)
    NT = time_horizon * time_multiplier
    if use_initial_enable:
        enable_constraints = initial_constraints.copy()
    else:
        enable_constraints = np.zeros(NL)

    all_model_time = 0
    all_solve_t = 0
    all_load_data_t = 0
    round_num = 1
    # 可以考虑每次把model传回去，然后在model的基础上修改
    largest_indices = []
    first_caculate = True
    model = 0

    while(True):
        model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, pch_start, pdis_start, model_time, solve_t, load_data_t, no_answer = \
            run_uc(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True,
                   use_solar=True,first_caculate=first_caculate, storage_data=storage_data, new_input=new_input, line_limit_rate=line_limit_rate,
                   solar_gen=solar_gen, enable_constraints=enable_constraints, u_start=u_start, y_start=y_start, z_start=z_start, largest_indices=largest_indices,
                   initial_model=0, fix_int = fix_int, time_limit=time_limit, mip_gap=mip_gap)
        if no_answer:
            return model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_model_time, all_solve_t, all_load_data_t, round_num, no_answer
        first_caculate = False
        all_model_time = all_model_time + model_time
        all_solve_t = all_solve_t + solve_t
        all_load_data_t = all_load_data_t + load_data_t

        l_value, NT, NL, normal_flow_limit = caculate_line_flow(data, p_start, use_storage=use_storage, use_new_input=use_new_input, storage_data=storage_data,
                                                                new_input=new_input, pch_start=pch_start, pdis_start=pdis_start)
        overload_percentage, largest_indices, is_overload, largest_values = get_overload_lines(l_value, normal_flow_limit, new_add_num, NT, NL)
        print(f"is_overload:{is_overload}, largest_values:{largest_values}, largest_indices:{largest_indices}")
        if is_overload:
            for index in largest_indices:
                i = index[0]
                initial_enable = enable_constraints[i]
                enable_constraints[i] = 1
            round_num = round_num + 1
        else:
            print(f"历经了{round_num}次循环")
            if add_more_constr:
                for i in range(NL):
                    if overload_percentage[i] > add_constr_gate:
                        enable_constraints[i] = 1

            return model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_value , enable_constraints, all_model_time, all_solve_t, all_load_data_t, round_num, no_answer

if __name__ == '__main__':
    path = 'C:\\Users\\94386\\.julia\\packages\\UnitCommitment\\G53EO\\instances\\'
    cases = [
        "matpower/case14/2017-01-01"
        "pglib-uc/ca/2014-09-01_reserves_0",
        "pglib-uc/ca/2014-09-01_reserves_1",
        "pglib-uc/ca/2015-03-01_reserves_0",
        "pglib-uc/ca/2015-06-01_reserves_0",
        "pglib-uc/ca/Scenario400_reserves_1",
        "pglib-uc/ferc/2015-01-01_lw",
        "pglib-uc/ferc/2015-05-01_lw",
        "pglib-uc/ferc/2015-07-01_hw",
        "pglib-uc/ferc/2015-10-01_lw",
        "pglib-uc/ferc/2015-12-01_lw",
        "pglib-uc/rts_gmlc/2020-04-03",
        "pglib-uc/rts_gmlc/2020-09-20",
        "pglib-uc/rts_gmlc/2020-10-27",
        "pglib-uc/rts_gmlc/2020-11-25",
        "pglib-uc/rts_gmlc/2020-12-23",
        "or-lib/20_0_1_w",
        "or-lib/20_0_5_w",
        "or-lib/50_0_2_w",
        "or-lib/75_0_2_w",
        "or-lib/100_0_1_w",
        "or-lib/100_0_4_w",
        "or-lib/100_0_5_w",
        "or-lib/200_0_3_w",
        "or-lib/200_0_7_w",
        "or-lib/200_0_9_w",
        "tejada19/UC_24h_290g",
        "tejada19/UC_24h_623g",
        "tejada19/UC_24h_959g",
        "tejada19/UC_24h_1577g",
        "tejada19/UC_24h_1888g",
        "tejada19/UC_168h_72g",
        "tejada19/UC_168h_86g",
        "tejada19/UC_168h_130g",
        "tejada19/UC_168h_131g",
        "tejada19/UC_168h_199g",
        "matpower/case1888rte/2017-02-01",
        "matpower/case1888rte/2017-01-01",
        "matpower/case1951rte/2017-02-01",
        "matpower/case2848rte/2017-02-01",
        "matpower/case2746wp/2017-02-01",
        "matpower/case2869pegase/2017-02-01",
        "matpower/case3012wp/2017-02-01",
        "matpower/case3375wp/2017-02-01",
        "matpower/case6468rte/2017-02-01",
        "matpower/case6515rte/2017-02-01",
        "matpower/case118/2017-02-01",
        "matpower/case300/2017-01-01",
        "matpower/case89pegase/2017-01-01",
        "matpower/case30/2017-02-01",
        "matpower/case1354pegase/2017-01-01"
    ]


    results = {}
    # for case in cases:
    # cases with active line limit constraints: case89pegase, case2848rte

    use_storage = True

    # 获取储能数据
    storage_data_name = 'storage_data_1888.mat'
    data_dict_name = 'training_data'
    storage_data = _load_storage_or_solar(os.path.join(data_dict_name, 'storage_data_1888.npz'))

    # 获取太阳能数据（未使用）
    solar_data = _load_storage_or_solar(os.path.join(data_dict_name, 'solar_gen.npz'))
    solar_curve = solar_data['load'][0]

    # 获取测试数据
    case = "matpower/case1888rte/2017-01-01"
    file = gzip.open(path + case + '.json.gz')
    data = json.loads(file.read())

    # 获取模型数据和训练数据
    # model_name_list = ['transformer_all_training_data_89pegase_data_not_random_1000_30000']
    # model_name = model_name_list[0]
    # dict_name = 'module_save'
    # model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(model_name, dict_name)
    # data_path = data_dict_name + '\\' + data_name
    # inputs, targets, test_inputs, test_targets, time_num, input_size, output_size, average_load = get_data(model_type, data_path,
    #                                                                                          test_num,
    #                                                                                          training_num,
    #                                                                                          data_is_random=data_is_random,
    #                                                                                          cutted_by_time=cutted_by_time)

    # 下一步读取数据，然后测试热启动效果(全部热启动和只热启动确定的那一部分，最好还是专门写一个文件来执行)

    time_horizon = data['Parameters'].get('Time horizon (h)') or data['Parameters'].get('Time (h)')
    time_step = data['Parameters'].get('Time step (min)', 60)
    assert(60 % time_step == 0)
    time_multiplier = int(60 / time_step)
    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])

    NT = time_horizon * time_multiplier
    NL = len(lines)

    # 增大峰谷差
    magnify_rate = 2
    line_limit_rate = 1
    solar_base_power = 3000
    solar_gen = get_solar_gen(solar_curve, solar_base_power)

    load_data = get_load(data)
    new_load_data = magnify_peak_valley_difference(load_data, magnify_rate)

    # 迭代添加约束，无热启动

    print("开始求解")
    start = time.time()
    model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, all_load_data_t, round_num, no_answer = \
        run_constraint_filter(data, is_warm_start=False, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True,
               use_solar= True, storage_data=storage_data, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen, new_add_num=5)
#    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
    end = time.time()
    print(f"求解结果为：{results}, have answer:{not no_answer}")
    print(f"总时间为：{end-start}, 建模时间为：{model_time}, solve_t为{solve_t},数据读取时间为{all_load_data_t},  round_num为{round_num}")
    model_time_no_warmstart = model_time
    total_time_no_warmstart = end - start
    solve_t_no_warmstart = solve_t
    load_data_t_no_warmstart = all_load_data_t

    true_constraints = enable_constraints

    # 开始热启动，此时热启动数值全部正确
#     start = time.time()
#     model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, all_load_data_t, round_num, no_answer = run_constraint_filter(data, is_warm_start=True, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True, storage_data=storage_data
#                                                 ,use_solar= True, u_start = u_start, y_start = y_start, z_start = z_start, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen, new_add_num=5)
# #    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
#     end = time.time()
#     print(f"求解结果为：{results}")
#     print(f"总时间为：{end-start}, 建模时间为：{model_time}, solve_t为{solve_t},数据读取时间为{all_load_data_t}, round_num为{round_num}")
#     model_time_all_right_warmstart = model_time
#     total_time_all_right_warmstart = end - start
#     solve_t_all_right_warmstart = solve_t
#     load_data_t_warmstart = all_load_data_t
#
#     true_constraints = enable_constraints

    # 无热启动，添加正确约束

    # start = time.time()
    # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, all_load_data_t, round_num, no_answer = run_constraint_filter(
    #     data, is_warm_start=False, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True,
    #     storage_data=storage_data
    #     , use_solar=True, use_initial_enable=True, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_load_data,
    #     line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=5, initial_constraints=true_constraints)
    # #    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
    # end = time.time()
    # print(f"求解结果为：{results}")
    # print(
    #     f"总时间为：{end - start}, 建模时间为：{model_time}, solve_t为{solve_t},数据读取时间为{all_load_data_t}, round_num为{round_num}")
    # model_time_true_constr = model_time
    # total_time_true_constr = end - start
    # solve_t_true_constr = solve_t
    # load_data_t_true_constr = all_load_data_t
    #
    # # 无热启动，添加正确约束
    #
    # all_constr = np.ones_like(true_constraints)
    #
    # start = time.time()
    # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, model_time, solve_t, all_load_data_t, round_num, no_answer = run_constraint_filter(
    #     data, is_warm_start=False, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True,
    #     storage_data=storage_data
    #     , use_solar=True, use_initial_enable=True, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_load_data,
    #     line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=5, initial_constraints=all_constr)
    # #    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
    # end = time.time()
    # print(f"求解结果为：{results}")
    # print(
    #     f"总时间为：{end - start}, 建模时间为：{model_time}, solve_t为{solve_t},数据读取时间为{all_load_data_t}, round_num为{round_num}")
    # model_time_all_constr = model_time
    # total_time_all_constr = end - start
    # solve_t_all_constr = solve_t
    # load_data_t_all_constr = all_load_data_t
    #
    # print(f"求解结果为：{results}")
    # # print(f"按照全部预测正确，热启动时，总时间为：{total_time_all_right_warmstart},建模时间为：{model_time_all_right_warmstart},数据读取时间为:{load_data_t_warmstart},  solve_t为{solve_t_all_right_warmstart}")
    # print(f"无热启动时,无初始约束时，总时间为：{total_time_no_warmstart},建模时间为：{model_time_no_warmstart},数据读取时间为:{load_data_t_no_warmstart},  solve_t为{solve_t_no_warmstart}")
    # print(f"无热启动时，初始约束全部正确时，总时间为：{total_time_true_constr},建模时间为：{model_time_true_constr},数据读取时间为:{load_data_t_true_constr},  solve_t为{solve_t_true_constr}")
    # print(f"无热启动时，初始添加全部约束时，总时间为：{total_time_all_constr},建模时间为：{model_time_all_constr},数据读取时间为:{load_data_t_all_constr},  solve_t为{solve_t_all_constr}")
