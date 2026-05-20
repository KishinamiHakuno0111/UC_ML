import os
import time
import gzip
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from ml_basic_functions import ml_load_model, get_data


def _load_array(path):
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io
    return scipy.io.loadmat(path)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


def get_solar_gen(solar_data, base_power):
    return solar_data * base_power


def try_get_from_list(x, t):
    return x[t] if type(x) == list else x


def timeseries(x, t):
    return x if type(x) == list else [x] * t


def map_list(input_list, map_func):
    return [map_func(value) for value in input_list]


def set_warmstart(x, x_start):
    key_list = x_start.keys()
    for k in key_list:
        x[k].start = x_start[k]

def get_optimize_data(data, use_new_min_up_down_time=False, use_new_input=False, use_storage=False, new_input=0):
    unit_names, units = zip(*data['Generators'].items())
    bus_names, buses = zip(*data['Buses'].items())
    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
    reserves = data['Reserves']
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
    assert (60 % time_step == 0)
    time_multiplier = int(60 / time_step)

    model = gp.Model()
    NT = time_horizon * time_multiplier
    NG = len(units)
    NB = len(buses)
    NL = len(lines)
    T_SET = range(NT)  # index set of time periods, start from 0
    G_SET = range(NG)
    B_SET = range(NB)
    L_SET = range(NL)

    B_TO_L_SET = [np.where(abs(ptdf[i, :]) > 0)[0].tolist() for i in L_SET]
    G_TO_L_SET = [[unit_idx for unit_idx in G_SET if bus_idx_map[units[unit_idx]['Bus']] in B_TO_L_SET[i]] for i in
                  L_SET]

    if use_storage:
        # 录入储能设施数据
        sto_max_power = storage_data['max_power'][0]
        sto_min_power = storage_data['min_power'][0]
        sto_initial_power = storage_data['initial_power'][0]
        sto_storage_position = storage_data['storage_position'][0]
        sto_Kbat = storage_data['Kbat'][0]
        sto_Eta = storage_data['Eta'][0]
        sto_p_ch_max = storage_data['p_ch_max'][0]
        sto_p_dis_max = storage_data['p_dis_max'][0]
        sto_num = len(sto_initial_power)

    # raw parameters
    must_run = map_list(units, lambda x: x.get('Must run?', False))
    provides_spinning_reserve = map_list(units, lambda x: x.get('Provides spinning reserves?', True))
    min_power = np.array(map_list(units, lambda x: timeseries(x['Production cost curve (MW)'][0], NT)))
    max_power = np.array(map_list(units, lambda x: timeseries(x['Production cost curve (MW)'][-1], NT)))
    initial_status = map_list(units, lambda x: x['Initial status (h)'] * time_multiplier)
    initial_power = np.array(map_list(units, lambda x: x['Initial power (MW)']))

    # 最小启停时间
    if use_new_min_up_down_time:
        min_uptime = [units[i].get('Minimum uptime (h)', 1) * time_multiplier for i in G_SET]
        min_uptime[2] = 6
        min_uptime[9] = 6
        min_downtime = [units[i].get('Minimum downtime (h)', 1) * time_multiplier for i in G_SET]
        min_downtime[2] = 6
        min_downtime[9] = 6
    else:
        min_uptime = map_list(units, lambda x: x.get('Minimum uptime (h)', 1) * time_multiplier)
        min_downtime = map_list(units, lambda x: x.get('Minimum downtime (h)', 1) * time_multiplier)

    ramp_up_limit = np.array(map_list(units, lambda x: x.get('Ramp up limit (MW)', GRB.INFINITY)))
    ramp_down_limit = np.array(map_list(units, lambda x: x.get('Ramp down limit (MW)', GRB.INFINITY)))
    startup_limit = np.array(map_list(units, lambda x: x.get('Startup limit (MW)', GRB.INFINITY)))
    shutdown_limit = np.array(map_list(units, lambda x: x.get('Shutdown limit (MW)', GRB.INFINITY)))
    startup_cost = map_list(units, lambda x: x.get('Startup costs ($)', [0.0])[0])

    piecewise_power = map_list(units,
                               lambda x: [[try_get_from_list(value, t) for value in x['Production cost curve (MW)']] for
                                          t in T_SET])  # NG * NT * NSEG
    piecewise_cost = map_list(units,
                              lambda x: [[try_get_from_list(value, t) for value in x['Production cost curve ($)']] for t
                                         in T_SET])  # NG * NT * NSEG

    if use_new_input:
        load_demand = new_input
    else:
        load_demand = np.array(map_list(buses, lambda x: timeseries(x['Load (MW)'], NT)))

    reserve_requirement = np.array(reserves['r1']['Amount (MW)'])
    LM = np.array(map_list(lines, lambda x: 1 if 'Normal flow limit (MW)' in x else 0))
    normal_flow_limit = np.array(map_list(lines, lambda x: x.get('Normal flow limit (MW)', 1e6))) * line_limit_rate

    bus_to_line_factor = [{j: ptdf[i, j] for j in B_TO_L_SET[i]} for i in L_SET]
    unit_to_line_factor = [{j: ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_TO_L_SET[i]} for i in L_SET]

    # line_flow_by_load = bus_to_line_factor @ load_demand
    init_u = np.array(map_list(initial_status, lambda x: 1 if x > 0 else 0))
    actual_startup_limit = np.minimum(np.reshape(startup_limit, (NG, 1)) @ np.ones((1, NT)),
                                      np.reshape(ramp_up_limit, (NG, 1)) @ np.ones((1, NT)) + min_power)
    actual_shutdown_limit = np.minimum(np.reshape(shutdown_limit, (NG, 1)) @ np.ones((1, NT)),
                                       np.reshape(ramp_down_limit, (NG, 1)) @ np.ones((1, NT)) + min_power)
    if use_storage:
        optimize_data = {'reserves':reserves, 'NT':NT, 'NB':NB, 'NG':NG, 'NL':NL, 'T_SET':T_SET, 'G_SET':G_SET, 'L_SET':L_SET,
                     'B_SET':B_SET, 'B_TO_L_SET':B_TO_L_SET, 'G_TO_L_SET':G_TO_L_SET, 'sto_max_power':sto_max_power, 'sto_min_power':sto_min_power,
                         'sto_initial_power':sto_initial_power, 'sto_storage_position': sto_storage_position, 'sto_Kbat':sto_Kbat,
                         'sto_Eta':sto_Eta, 'sto_p_ch_max':sto_p_ch_max, 'sto_p_dis_max':sto_p_dis_max, 'sto_num':sto_num,
                         'must_run': must_run, 'provides_spinning_reserve':provides_spinning_reserve, 'min_power':min_power,
                         'max_power':max_power, 'initial_status':initial_status, 'initial_power':initial_power, 'min_uptime':min_uptime,
                         'min_down_time':min_downtime, 'ramp_up_limit':ramp_up_limit, 'ramp_down_limit':ramp_down_limit,
                         'startup_limit':startup_limit, 'shutdown_limit':shutdown_limit, 'startup_cost':startup_cost,
                         'piecewise_power':piecewise_power, 'piecewise_cost':piecewise_cost, 'load_demand':load_demand,
                         'reserve_requirement':reserve_requirement, 'LM':LM, 'normal_flow_limit':normal_flow_limit,
                         'bus_to_line_factor':bus_to_line_factor, 'unit_to_line_factor':unit_to_line_factor, 'init_u': init_u,
                         'actual_startup_limit': actual_startup_limit, 'actual_shutdown_limit': actual_shutdown_limit}
    else:
        optimize_data = {'reserves':reserves, 'NT':NT, 'NB':NB, 'NG':NG, 'NL':NL, 'T_SET':T_SET, 'G_SET':G_SET, 'L_SET':L_SET,
                     'B_SET':B_SET, 'B_TO_L_SET':B_TO_L_SET, 'G_TO_L_SET':G_TO_L_SET, }
    return optimize_data


def run_uc(data, is_warm_start = False, use_new_input = False, use_storage = False, use_new_min_up_down_time = False, first_caculate=True,
           use_solar = False ,u_start = 0, y_start = 0, z_start = 0, new_input = 0, storage_data = 0, line_limit_rate = 1, solar_gen = 0, LPwarmstart=2, enable_constraints = 0,
           largest_indices=0, initial_model=0, fix_int = False, time_limit = 20, mip_gap = 1e-3):

    load_data_start_time = time.time()
    # refer to https://anl-ceeesa.github.io/UnitCommitment.jl/0.2/format/ for formats
    # note that version 0.2.0 is required, newer versions are not supported
    unit_names, units = zip(*data['Generators'].items())
    bus_names, buses = zip(*data['Buses'].items())
    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
    reserves = data['Reserves']
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

    model = gp.Model()
    NT = time_horizon * time_multiplier
    NG = len(units)
    NB = len(buses)
    NL = len(lines)
    T_SET = range(NT)  # index set of time periods, start from 0
    G_SET = range(NG)
    B_SET = range(NB)
    L_SET = range(NL)

    # 这里
    # B_TO_L_SET = [np.where(abs(ptdf[i, :]) > 0)[0].tolist() for i in L_SET]
    # G_TO_L_SET = [[unit_idx for unit_idx in G_SET if bus_idx_map[units[unit_idx]['Bus']] in B_TO_L_SET[i]] for i in L_SET] # 好慢

    if use_storage:
        # 录入储能设施数据
        if len(storage_data['max_power']) != 1 :
            sto_max_power = storage_data['max_power']
            sto_min_power = storage_data['min_power']
            sto_initial_power = storage_data['initial_power']
            sto_storage_position = storage_data['storage_position']
            sto_Kbat = storage_data['Kbat']
            sto_Eta = storage_data['Eta']
            sto_p_ch_max = storage_data['p_ch_max']
            sto_p_dis_max = storage_data['p_dis_max']
            sto_num = len(sto_initial_power)
        else:
            sto_max_power = storage_data['max_power'][0]
            sto_min_power = storage_data['min_power'][0]
            sto_initial_power = storage_data['initial_power'][0]
            sto_storage_position = storage_data['storage_position'][0]
            sto_Kbat = storage_data['Kbat'][0]
            sto_Eta = storage_data['Eta'][0]
            sto_p_ch_max = storage_data['p_ch_max'][0]
            sto_p_dis_max = storage_data['p_dis_max'][0]
            sto_num = len(sto_initial_power)

        # 设置变量
        pch = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="pch")
        pdis = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="pdis")
        stopower = model.addVars(sto_num, NT, vtype=GRB.CONTINUOUS, name="stopower")
        sto_c = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="sto_c")

    # raw parameters
    must_run = map_list(units, lambda x: x.get('Must run?', False))
    provides_spinning_reserve = map_list(units, lambda x: x.get('Provides spinning reserves?', True))
    min_power = np.array(map_list(units, lambda x: timeseries(x['Production cost curve (MW)'][0], NT)))
    max_power = np.array(map_list(units, lambda x: timeseries(x['Production cost curve (MW)'][-1], NT)))
    initial_status = map_list(units, lambda x: x['Initial status (h)'] * time_multiplier)
    initial_power = np.array(map_list(units, lambda x: x['Initial power (MW)']))

    # 最小启停时间
    if use_new_min_up_down_time:
        min_uptime = [units[i].get('Minimum uptime (h)', 1) * time_multiplier for i in G_SET]
        # min_uptime[2] = 6
        # min_uptime[9] = 6
        min_downtime = [units[i].get('Minimum downtime (h)', 1) * time_multiplier for i in G_SET]
        # min_downtime[2] = 6
        # min_downtime[9] = 6
    else:
        min_uptime = map_list(units, lambda x: x.get('Minimum uptime (h)', 1) * time_multiplier)
        min_downtime = map_list(units, lambda x: x.get('Minimum downtime (h)', 1) * time_multiplier)

    ramp_up_limit = np.array(map_list(units, lambda x: x.get('Ramp up limit (MW)', GRB.INFINITY)))
    ramp_down_limit = np.array(map_list(units, lambda x: x.get('Ramp down limit (MW)', GRB.INFINITY)))
    startup_limit = np.array(map_list(units, lambda x: x.get('Startup limit (MW)', GRB.INFINITY)))
    shutdown_limit = np.array(map_list(units, lambda x: x.get('Shutdown limit (MW)', GRB.INFINITY)))
    startup_cost = map_list(units, lambda x: x.get('Startup costs ($)', [0.0])[0])

    piecewise_power = map_list(units, lambda x: [[try_get_from_list(value, t) for value in x['Production cost curve (MW)']] for t in T_SET])  # NG * NT * NSEG
    piecewise_cost = map_list(units, lambda x: [[try_get_from_list(value, t) for value in x['Production cost curve ($)']] for t in T_SET])  # NG * NT * NSEG

    if use_new_input:
        load_demand = new_input
    else:
        load_demand = np.array(map_list(buses, lambda x: timeseries(x['Load (MW)'], NT)))

    reserve_requirement = np.array(reserves['r1']['Amount (MW)'])
    LM = np.array(map_list(lines, lambda x: 1 if 'Normal flow limit (MW)' in x else 0))
    normal_flow_limit = np.array(map_list(lines, lambda x: x.get('Normal flow limit (MW)', 1e6))) * line_limit_rate

    # 这里
    # bus_to_line_factor = [{j: ptdf[i, j] for j in B_TO_L_SET[i]} for i in L_SET]
    # unit_to_line_factor = [{j: ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_TO_L_SET[i]} for i in L_SET]

    # line_flow_by_load = bus_to_line_factor @ load_demand
    init_u = np.array(map_list(initial_status, lambda x: 1 if x > 0 else 0))
    actual_startup_limit = np.minimum(np.reshape(startup_limit, (NG, 1)) @ np.ones((1, NT)), np.reshape(ramp_up_limit, (NG, 1)) @ np.ones((1, NT)) + min_power)
    actual_shutdown_limit = np.minimum(np.reshape(shutdown_limit, (NG, 1)) @ np.ones((1, NT)), np.reshape(ramp_down_limit, (NG, 1)) @ np.ones((1, NT)) + min_power)

    unit_to_line_matrix = np.array([[ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_SET] for i in L_SET])
    sto_to_line_matrix = np.array([[ptdf[i, sto_storage_position[j]] for j in range(sto_num)] for i in L_SET])

    load_data_end_time = time.time()

    print(f"读取数据花费时间为{load_data_end_time-load_data_start_time}")
    load_data_time = load_data_end_time-load_data_start_time
    model_start_time = time.time()


    # 定义变量
    u = model.addVars(NG, NT, vtype=GRB.BINARY, name="u")  # is_on
    y = model.addVars(NG, NT, vtype=GRB.BINARY, name="y")  # switch_on
    z = model.addVars(NG, NT, vtype=GRB.BINARY, name="z")  # switch_off
    p = model.addVars(NG, NT, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="p")  # power
    r = model.addVars(NG, NT, lb=0, vtype=GRB.CONTINUOUS, name="r")  # spinning reserve
    c = model.addVars(NG, NT, lb=0, vtype=GRB.CONTINUOUS, name="c")  # piecewise linear cost
    l = model.addVars(NL, NT, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="l")  # line flow
    l_punish = model.addVars(NL, NT, lb=0, vtype=GRB.CONTINUOUS, name="l_punish")  # line flow
    flow_equal_punish = model.addVars(NT, lb=0, vtype=GRB.CONTINUOUS, name="flow_equal_punish")
    # model.setParam('Start', {})

    if use_storage:
        # 设置变量
        pch = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="pch")
        pdis = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="pdis")
        stopower = model.addVars(sto_num, NT, vtype=GRB.CONTINUOUS, name="stopower")
        sto_c = model.addVars(sto_num, NT, lb=0, vtype=GRB.CONTINUOUS, name="sto_c")

    # 设置热启动
    if is_warm_start:
        set_warmstart(u, u_start)
        set_warmstart(y, y_start)
        set_warmstart(z, z_start)
        # for k in u_start.keys():
        #     u[k].start = u_start[k]
        # for k in y_start.keys():
        #     y[k].start = y_start[k]
        # for k in z_start.keys():
        #     z[k].start = z_start[k]
        # model.Params.LPWarmStart = LPwarmstart


    # define variables
    if fix_int:
        u_keys = u_start.keys()
        y_keys = y_start.keys()
        z_keys = z_start.keys()

        for k in u_keys:
            u[k[0], k[1]].lb = u_start[k]
            u[k[0], k[1]].ub = u_start[k]

        for k in y_keys:
            y[k[0], k[1]].lb = y_start[k]
            y[k[0], k[1]].ub = y_start[k]
        for k in z_keys:
            z[k[0], k[1]].lb = z_start[k]
            z[k[0], k[1]].ub = z_start[k]


    # fix variables
    for gi in G_SET:
        if must_run[gi]:
            u[gi, :].lb = 1
            y[gi, :].ub = 0
            z[gi, :].ub = 0
            if initial_status[gi] < 0:
                y[gi, :].lb = 1
        if not provides_spinning_reserve[gi]:
            r[gi, :].ub = 0
    if use_storage:
        # 固定储能装置的初始储能量
        for i in range(sto_num):
            stopower[i, 0].ub = sto_initial_power[i]
            stopower[i, 0].lb = sto_initial_power[i]

    for i in G_SET:
        if must_run[i]:
            continue
        # 初始最小启停时间约束
        if initial_status[i] > 0 and min_uptime[i] > initial_status[i]:
            model.addLConstr(lhs=gp.quicksum(z[i, t] for t in range(0, min(min_uptime[i]-initial_status[i], NT+1))), sense=GRB.EQUAL, rhs=0, name=f'init_min_uptime_f{i}')
        if initial_status[i] < 0 and min_downtime[i] > -initial_status[i]:
            model.addLConstr(lhs=gp.quicksum(y[i, t] for t in range(0, min(min_downtime[i] + initial_status[i], NT+1))), sense=GRB.EQUAL, rhs=0, name=f'init_min_downtime_f{i}')
        for t in T_SET:
            # 最小启停时间约束
            model.addLConstr(lhs=gp.quicksum(y[i, t2] for t2 in range(max(t - min_uptime[i] + 1, 0), t + 1)),
                             sense=GRB.LESS_EQUAL, rhs=u[i, t], name=f'min_uptime_{i}_{t}')
            model.addLConstr(lhs=gp.quicksum(z[i, t2] for t2 in range(max(t - min_uptime[i] + 1, 0), t + 1)),
                             sense=GRB.LESS_EQUAL, rhs=1 - u[i, t], name=f'min_downtime_{i}_{t}')

    for i in G_SET:
        # 初始时刻的逻辑约束
        model.addLConstr(lhs=u[i, 0]-init_u[i], sense=GRB.EQUAL, rhs=y[i, 0]-z[i, 0], name=f'logic0_{i}')

        # 初始时刻的爬坡约束
        model.addLConstr(lhs=p[i, 0]-initial_power[i]+r[i, 0], sense=GRB.LESS_EQUAL, rhs=ramp_up_limit[i]*init_u[i]+actual_startup_limit[i, 0]*y[i, 0], name=f'ramp_up0_{i}')
        model.addLConstr(lhs=initial_power[i]-p[i, 0], sense=GRB.LESS_EQUAL, rhs=ramp_down_limit[i]*init_u[i]+actual_shutdown_limit[i, 0]*z[i, 0], name=f'ramp_down0_{i}')

        for t in T_SET:
            # 不能同时启停
            model.addLConstr(lhs=y[i, t]+z[i, t], sense=GRB.LESS_EQUAL, rhs=1, name=f'logic_rule_{i}_{t}')

            # 最大最小功率约束
            model.addLConstr(lhs=p[i, t], sense=GRB.GREATER_EQUAL, rhs=u[i, t]*min_power[i, t], name=f'min_power_{i}_{t}')
            model.addLConstr(lhs=p[i, t]+r[i, t], sense=GRB.LESS_EQUAL, rhs=u[i, t] * max_power[i, t], name=f'max_power_{i}_{t}')

    for i in G_SET:
        for t in range(1, NT):
            # 逻辑约束
            model.addLConstr(lhs=u[i, t]-u[i, t-1], sense=GRB.EQUAL, rhs=y[i, t]-z[i, t], name=f'logic_{i}_{t}')

            # 爬坡约束
            model.addLConstr(lhs=p[i, t]-p[i, t-1]+r[i, t], sense=GRB.LESS_EQUAL, rhs=ramp_up_limit[i]*u[i, t-1]+actual_startup_limit[i, t]*y[i, t], name=f'ramp_up_{i}_{t}')
            model.addLConstr(lhs=p[i, t-1]-p[i, t], sense=GRB.LESS_EQUAL, rhs=ramp_down_limit[i] * u[i, t] + actual_shutdown_limit[i, t] * z[i, t], name=f'ramp_down_{i}_{t}')

    for t in T_SET:
        # 预备容量约束
        model.addLConstr(lhs=gp.quicksum(r[i, t] for i in G_SET), sense=GRB.GREATER_EQUAL, rhs=reserve_requirement[t], name=f'reserve_{t}')

    if use_storage:
        for i in L_SET:
            if normal_flow_limit[i] >= 1e6:
                continue
            # 记得后面逐步添加约束时也要去掉那些实际没有约束的线路
            if enable_constraints[i] > 0.5:
                for t in T_SET:
                    # 线路潮流约束
                    model.addLConstr(lhs=l[i, t], sense=GRB.EQUAL, rhs=gp.quicksum(
                        unit_to_line_matrix[i, j] * p[j, t] for j in G_SET) - gp.quicksum(
                        ptdf[i, j] * load_demand[j, t] for j in B_SET) + gp.quicksum(
                        sto_to_line_matrix[i, j] * (pdis[j, t] - pch[j, t]) for j in range(sto_num)),
                                     name=f'line_flow_rule_{i}_{t}')
                    model.addLConstr(lhs=l[i, t]-l_punish[i, t], sense=GRB.LESS_EQUAL, rhs=normal_flow_limit[i]-1e-3,
                                     name=f'line_flow_upper_bound_{i}_{t}')
                    model.addLConstr(lhs=l[i, t]+l_punish[i, t], sense=GRB.GREATER_EQUAL, rhs=-normal_flow_limit[i]+1e-3,
                                     name=f'line_flow_lower_bound_{i}_{t}')
    else:
        for i in L_SET:
            if normal_flow_limit[i] >= 1e6:
                continue
            # 记得后面逐步添加约束时也要去掉那些实际没有约束的线路
            if enable_constraints[i] > 0.5:
                for t in T_SET:
                    # 线路潮流约束
                    model.addLConstr(lhs=l[i, t], sense=GRB.EQUAL, rhs=gp.quicksum(
                        unit_to_line_matrix[i, j] * p[j, t] for j in G_SET) - gp.quicksum(
                        ptdf[i, j] * load_demand[j, t] for j in B_SET),
                                     name=f'line_flow_rule_{i}_{t}')
                    model.addLConstr(lhs=l[i, t], sense=GRB.LESS_EQUAL, rhs=normal_flow_limit[i]-1e-3,
                                     name=f'line_flow_upper_bound_{i}_{t}')
                    model.addLConstr(lhs=l[i, t], sense=GRB.GREATER_EQUAL, rhs=-normal_flow_limit[i]+1e-3,
                                     name=f'line_flow_lower_bound_{i}_{t}')

    if use_storage:
        for t in T_SET:
            # 功率平衡约束
            model.addLConstr(lhs=gp.quicksum(p[i, t] for i in G_SET) + gp.quicksum(
            (pdis[i, t]-pch[i, t]) for i in range(sto_num)) + flow_equal_punish[t], sense=GRB.EQUAL,
                             rhs=gp.quicksum(load_demand[j, t] for j in B_SET), name=f'power_balance_{t}')
    else:
        for t in T_SET:
            # 功率平衡约束
            model.addLConstr(lhs=gp.quicksum(p[i, t] for i in G_SET) + flow_equal_punish[t], sense=GRB.EQUAL,
                             rhs=gp.quicksum(load_demand[j, t] for j in B_SET), name=f'power_balance_{t}')

    if use_storage:
        for i in range(sto_num):
            for t in T_SET:
                # 充放电上限约束
                model.addLConstr(lhs=pch[i, t], sense=GRB.LESS_EQUAL, rhs=sto_p_ch_max[i], name=f'charge_upper_bound_{i}_{t}')
                model.addLConstr(lhs=pdis[i, t], sense=GRB.LESS_EQUAL, rhs=sto_p_dis_max[i], name=f'discharge_upper_bound_{i}_{t}')

                #储能上下限
                model.addLConstr(lhs=stopower[i, t], sense=GRB.LESS_EQUAL, rhs=sto_max_power[i], name=f'storage_max_{i}_{t}')
                model.addLConstr(lhs=stopower[i, t], sense=GRB.GREATER_EQUAL, rhs=sto_min_power[i], name=f'storage_min_{i}_{t}')
                model.addLConstr(lhs=sto_c[i, t], sense=GRB.EQUAL, rhs=sto_Kbat[i]*(pdis[i, t]/sto_Eta[i]+pch[i, t]*sto_Eta[i]), name=f'storage_cost_{i}_{t}')

            for t in range(1, NT):
                # 容量逻辑约束
                model.addLConstr(lhs=stopower[i, t], sense=GRB.EQUAL, rhs=stopower[i, t-1]+sto_Eta[i]*pch[i, t-1]-(1/sto_Eta[i])*pdis[i, t-1], name=f'capacity_logic_rule_{i}_{t}')


    # piecewise cost not considering u
    # for i in G_SET:
    #     for t in T_SET:
    #         if len(piecewise_cost[i][t]) > 1:
    #             model.addGenConstrPWL(p[i, t], c[i, t], piecewise_power[i][t], piecewise_cost[i][t], name='cost_' + str(i) + '_' + str(t))
    #         else:
    #             model.addConstr(c[i, t] == u[i, t] * piecewise_cost[i][t][0], name='cost_' + str(i) + '_' + str(t))

    # 设定目标函数
    n_seg = max(len(piecewise_cost[i][t]) for i in G_SET for t in T_SET)

    for i in G_SET:
        for t in T_SET:
            for i_seg in range(max(1, n_seg - 1)):
                if i_seg+1 < len(piecewise_cost[i][t]):
                    if abs(piecewise_power[i][t][i_seg + 1] - piecewise_power[i][t][i_seg]) < 1e-6:
                        slope = 0
                    else:
                        slope = (piecewise_cost[i][t][i_seg + 1] - piecewise_cost[i][t][i_seg]) / (
                                piecewise_power[i][t][i_seg + 1] - piecewise_power[i][t][i_seg])
                    model.addLConstr(lhs=c[i, t]-u[i, t]*piecewise_cost[i][t][i_seg], sense=GRB.GREATER_EQUAL, rhs=(p[i, t] - u[i, t] * piecewise_power[i][t][i_seg]) * slope, name=f'cost_constr_{i}_{t}')
                elif i_seg == 0 and len(piecewise_cost[i][t]) == 1:
                    # units not adjustable
                    model.addLConstr(lhs=c[i, t]-u[i, t]*piecewise_cost[i][t][i_seg], sense=GRB.GREATER_EQUAL, rhs=0, name=f'cost_constr_{i}_{t}')
                else:
                    continue

    punish_C = 1e20
    power_equal_punish_C = 1e20
    if use_storage:
        model.setObjective(gp.quicksum(c[i, t]+startup_cost[i]*y[i, t] for i in G_SET for t in T_SET) + gp.quicksum(sto_c[i, t] for i in range(sto_num) for t in T_SET) + punish_C*gp.quicksum(l_punish[i, t] for i in L_SET for t in T_SET) + power_equal_punish_C*gp.quicksum(flow_equal_punish[t] for t in T_SET))
    else:
        model.setObjective(gp.quicksum(c[i, t]+startup_cost[i]*y[i, t] for i in G_SET for t in T_SET) + punish_C*gp.quicksum(l_punish[i, t] for i in L_SET for t in T_SET) + power_equal_punish_C*gp.quicksum(flow_equal_punish[t] for t in T_SET))
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print('model time: ' + str(model_time))

    # def mycallback(model, where):
    #     if where == GRB.Callback.MIPSOL:
    #         val_p = model.cbGetSolution(p)
    #         val_l = unit_to_line_factor @ val_p - line_flow_by_load
    #         upper_violated_index = np.where(val_l > np.reshape(normal_flow_limit, (NL, 1)))
    #         lower_violated_index = np.where(val_l < -np.reshape(normal_flow_limit, (NL, 1)))
    #         for i in range(len(upper_violated_index[0])):
    #             line_index = upper_violated_index[0][i]
    #             time_index = upper_violated_index[1][i]
    #             model.cbLazy((unit_to_line_factor[line_index, :] @ p[:, time_index]).item() - line_flow_by_load[line_index, time_index] <= normal_flow_limit[line_index])
    #         for i in range(len(lower_violated_index[0])):
    #             line_index = lower_violated_index[0][i]
    #             time_index = lower_violated_index[1][i]
    #             model.cbLazy((unit_to_line_factor[line_index, :] @ p[:, time_index]).item() - line_flow_by_load[line_index, time_index] >= -normal_flow_limit[line_index])

    # solve
    # model.Params.lazyConstraints = 1
    solve_start = time.time()
    output_flag = int(os.environ.get('UC_GUROBI_OUTPUTFLAG', '1'))
    model.setParam('OutputFlag', output_flag)
    threads_env = os.environ.get('UC_GUROBI_THREADS')
    if threads_env is not None and threads_env != '':
        model.setParam('Threads', int(threads_env))
    model.setParam('MIPGap', mip_gap)
    # model.setParam('TimeLimit', time_limit)
    model.optimize()
    solve_end = time.time()
    solve_t = -solve_start + solve_end
    print(f"求解时间为{solve_t}")

    u_start = {(i, t): round(u[i, t].x) for i in G_SET for t in T_SET}
    y_start = {(i, t): round(y[i, t].x) for i in G_SET for t in T_SET}
    z_start = {(i, t): round(z[i, t].x) for i in G_SET for t in T_SET}
    p_start = {(i, t): p[i, t].x for i in G_SET for t in T_SET}
    r_start = {(i, t): r[i, t].x for i in G_SET for t in T_SET}
    c_start = {(i, t): c[i, t].x for i in G_SET for t in T_SET}
    l_start = {(i, t): l[i, t].x for i in L_SET for t in T_SET}

    l_punish_matrix = np.array([[l_punish[i, t].x for t in T_SET] for i in L_SET])
    power_equal_punish = np.array([flow_equal_punish[t].x for t in T_SET])
    punish_max = np.max(l_punish_matrix, axis=(0, 1)) + np.max(power_equal_punish)
    no_answer = (punish_max > 1e-3)

    obj = model.getObjective()
    # model.setParam('MIPGap', 1e-3)
    result = obj.getValue()
    print(f"优化的结果为：{obj.getValue()}")
    print(
        f'有潮流越限的结果为：{True in [False in [-normal_flow_limit[i] <= l[i, t].x <= normal_flow_limit[i] for i in L_SET] for t in T_SET]}')

    if use_storage:
        pch_start = {(i, t): pch[i, t].x for i in range(sto_num) for t in T_SET}
        pdis_start = {(i, t): pdis[i, t].x for i in range(sto_num) for t in T_SET}
        return model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, pch_start, pdis_start, model_time, solve_t, load_data_time, no_answer
    else:
        return model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, model_time, solve_t, load_data_time, no_answer


if __name__ == '__main__':
    path = 'case_data/'
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
    storage_data_name = 'storage_data.mat'
    data_dict_name = 'training_data'
    storage_data = _load_array(os.path.join(data_dict_name, 'storage_data.mat'))

    # 获取太阳能数据（未使用）
    solar_data = _load_array(os.path.join(data_dict_name, 'solar_gen'))
    solar_curve = solar_data['load'][0]

    # 获取测试数据
    case = "matpower/case89pegase/2017-01-01"
    file = gzip.open(path + case + '.json.gz')
    data = json.loads(file.read())

    # 获取模型数据和训练数据
    # model_name_list = ['transformer_all_training_data_89pegase_data_not_random_1000_30000']
    # model_name = model_name_list[0]
    # dict_name = 'module_save'
    # model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name = ml_load_model(model_name, dict_name)
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
    enable_constraints = np.ones([NL])

    # 增大峰谷差
    magnify_rate = 1.05
    line_limit_rate = 1
    solar_base_power = 3000
    solar_gen = get_solar_gen(solar_curve, solar_base_power)

    load_data = get_load(data)
    new_load_data = magnify_peak_valley_difference(load_data, magnify_rate)

    print("开始求解")
    start = time.time()
    model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, model_time, solve_t = \
        run_uc(data, is_warm_start=False, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True,
               use_solar= True, storage_data=storage_data, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen, enable_constraints=enable_constraints)
#    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
    end = time.time()
    print(f"求解结果为：{results}")
    print(f"建模时间为：{model_time}, 求解时间为：{-model_time + end - start}, 总时间为：{end-start}, solve_t为{solve_t}")
    model_time_no_warmstart = model_time
    solve_time_no_warmstart = -model_time + end - start
    total_time_no_warmstart = end - start
    solve_t_no_warmstart = solve_t

    # 开始热启动，此时热启动数值全部正确
    start = time.time()
    model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, model_time, solve_t = run_uc(data, is_warm_start=True, use_new_input=True, use_storage=use_storage, use_new_min_up_down_time=True, storage_data=storage_data
                                                ,use_solar= True, u_start = u_start, y_start = y_start, z_start = z_start, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen, enable_constraints=enable_constraints)
#    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
    end = time.time()
    print(f"求解结果为：{results}")
    print(f"建模时间为：{model_time}, 求解时间为：{-model_time + end - start}, 总时间为：{end-start}, solve_t为{solve_t}")
    model_time_all_right_warmstart = model_time
    solve_time_all_right_warmstart = -model_time + end - start
    total_time_all_right_warmstart = end - start
    solve_t_all_right_warmstart = solve_t

    print(f"求解结果为：{results}")
    print(f"按照全部预测正确，热启动时，建模时间为：{model_time_all_right_warmstart}, 求解时间为：{solve_time_all_right_warmstart}, 总时间为：{total_time_all_right_warmstart}, solve_t为{solve_t_all_right_warmstart}")
    print(f"无热启动时，建模时间为：{model_time_no_warmstart}, 求解时间为：{solve_time_no_warmstart}, 总时间为：{total_time_no_warmstart}, solve_t为{solve_t_no_warmstart}")
