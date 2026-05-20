import os
import time
import gzip
import json
import numpy as np
import pyomo.environ as pyo
from pyomo.core import Param
import matplotlib.pyplot as plt
from ml_basic_functions import ml_load_model, get_data


def _load_array(path):
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io
    return scipy.io.loadmat(path)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 按照最高值增大rate倍，最小值减小rate倍来线性改变
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


def run_uc(data, is_warm_start = False, use_new_input = False, use_storage = False, use_new_min_up_down_time = False,
           use_solar = False ,u_start = 0, y_start = 0, z_start = 0, new_input = 0, storage_data = 0, line_limit_rate = 1, solar_gen = 0, fix_int = False, time_limit = 1e6, mip_gap = 1e-3):
    # refer to https://anl-ceeesa.github.io/UnitCommitment.jl/0.2/format/ for formats
    # note that version 0.2.0 is required, newer versions are not supported
    unit_names, units = zip(*data['Generators'].items())
    bus_names, buses = zip(*data['Buses'].items())
    line_names, lines = zip(*data['Transmission lines'].items()) if 'Transmission lines' in data else ([], [])
    reserves = data['Reserves']
    keys_of_reserves = reserves.keys()

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

    model = pyo.ConcreteModel()
    model.NT = time_horizon * time_multiplier
    model.NG = len(units)
    model.NB = len(buses)
    model.NL = len(lines)
    model.T_SET = pyo.Set(initialize=range(model.NT))  # index set of time periods, start from 0
    model.G_SET = pyo.Set(initialize=range(model.NG))
    model.B_SET = pyo.Set(initialize=range(model.NB))
    model.L_SET = pyo.Set(initialize=range(model.NL))
    # model.B_TO_L_SET = pyo.Set(model.L_SET, initialize=lambda m, i: np.where(abs(ptdf[i, :]) > 0)[0].tolist()) #  这一句很慢
    # model.G_TO_L_SET = pyo.Set(model.L_SET, initialize=lambda m, i: [unit_idx for unit_idx in m.G_SET if bus_idx_map[units[unit_idx]['Bus']] in m.B_TO_L_SET[i]]) #  这一句很慢
    B_TO_L_SET = [np.where(abs(ptdf[i, :]) > 0)[0].tolist() for i in model.L_SET]
    G_TO_L_SET = [[unit_idx for unit_idx in model.G_SET if bus_idx_map[units[unit_idx]['Bus']] in B_TO_L_SET[i]] for i in model.L_SET]

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

        # 设置变量
        model.pch = pyo.Var(range(sto_num), model.T_SET, bounds=(0, None))
        model.pdis = pyo.Var(range(sto_num), model.T_SET, bounds=(0, None))
        model.stopower = pyo.Var(range(sto_num), model.T_SET) # 储能的量
        model.sto_c = pyo.Var(range(sto_num), model.T_SET, bounds=(0, None)) # 储能成本

    model.must_run = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Must run?', False))
    model.provides_spinning_reserve = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Provides spinning reserves?', True))
    model.min_power = pyo.Param(model.G_SET, model.T_SET, initialize=lambda m, i, t: try_get_from_list(units[i]['Production cost curve (MW)'][0], t))
    model.max_power = pyo.Param(model.G_SET, model.T_SET, initialize=lambda m, i, t: try_get_from_list(units[i]['Production cost curve (MW)'][-1], t))
    model.initial_status = pyo.Param(model.G_SET, initialize=lambda m, i: units[i]['Initial status (h)'] * time_multiplier)
    model.initial_power = pyo.Param(model.G_SET, initialize=lambda m, i: units[i]['Initial power (MW)'])

    # 最小启停时间
    if use_new_min_up_down_time:
        min_uptime = [units[i].get('Minimum uptime (h)', 1) * time_multiplier for i in model.G_SET]
        min_uptime[2] = 6
        min_uptime[9] = 6
        min_downtime = [units[i].get('Minimum downtime (h)', 1) * time_multiplier for i in model.G_SET]
        min_downtime[2] = 6
        min_downtime[9] = 6
        model.min_uptime = pyo.Param(model.G_SET, initialize=lambda m, i: min_uptime[i])
        model.min_downtime = pyo.Param(model.G_SET, initialize=lambda m, i: min_downtime[i])
    else:
        model.min_uptime = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Minimum uptime (h)', 1) * time_multiplier)
        model.min_downtime = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Minimum downtime (h)', 1) * time_multiplier)

    model.ramp_up_limit = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Ramp up limit (MW)', 1e6))
    model.ramp_down_limit = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Ramp down limit (MW)', 1e6))
    model.startup_limit = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Startup limit (MW)', 1e6))
    model.shutdown_limit = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Shutdown limit (MW)', 1e6))
    model.startup_cost = pyo.Param(model.G_SET, initialize=lambda m, i: units[i].get('Startup costs ($)', [0.0])[0])

    model.piecewise_power = pyo.Param(model.G_SET, model.T_SET, domain=pyo.Any, initialize=lambda m, i, t: [try_get_from_list(value, t) for value in units[i]['Production cost curve (MW)']])
    model.piecewise_cost = pyo.Param(model.G_SET, model.T_SET, domain=pyo.Any, initialize=lambda m, i, t: [try_get_from_list(value, t) for value in units[i]['Production cost curve ($)']])
    if use_new_input:
        model.load_demand = pyo.Param(model.B_SET, model.T_SET, initialize=lambda m, i, t: new_input[i][t])
    else:
        model.load_demand = pyo.Param(model.B_SET, model.T_SET, initialize=lambda m, i, t: try_get_from_list(buses[i]['Load (MW)'], t))
    model.reserve_requirement: Param = pyo.Param(model.T_SET, initialize=lambda m, t: reserves['r1']['Amount (MW)'][t] if 'Amount (MW)' in reserves['r1'] else 0)
    model.LM = pyo.Param(model.L_SET, initialize=lambda m, i: 1 if 'Normal flow limit (MW)' in lines[i] else 0)

    model.normal_flow_limit = pyo.Param(model.L_SET, initialize=lambda m, i: line_limit_rate * lines[i].get('Normal flow limit (MW)', 1e6))

    # model.bus_to_line_factor = pyo.Param(model.L_SET, domain=pyo.Any, initialize=lambda m, i: {j: ptdf[i, j] for j in B_TO_L_SET[i]})
    # model.unit_to_line_factor = pyo.Param(model.L_SET, domain=pyo.Any, initialize=lambda m, i: {j: ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_TO_L_SET[i]})
    bus_to_line_factor = [{j: ptdf[i, j] for j in B_TO_L_SET[i]} for i in model.L_SET]
    unit_to_line_factor = [{j: ptdf[i, bus_idx_map[units[j]['Bus']]] for j in G_TO_L_SET[i]} for i in model.L_SET]

    model_start_time = time.time()

    # define variables
    if fix_int:
        model.u = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # is_on
        model.y = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # switch_on
        model.z = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # switch_off
        model.p = pyo.Var(model.G_SET, model.T_SET)  # power
        model.r = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # spinning reserve
        model.c = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # piecewise linear cost
        model.l = pyo.Var(model.L_SET, model.T_SET)

        u_keys = u_start.keys()
        y_keys = y_start.keys()
        z_keys = z_start.keys()

        for k in u_keys:
            model.u[k[0], k[1]].fix(u_start[k])

        for k in y_keys:
            model.y[k[0], k[1]].fix(y_start[k])

        for k in z_keys:
            model.z[k[0], k[1]].fix(z_start[k])


    elif not is_warm_start: # 若没有热启动
        model.u = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # is_on
        model.y = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # switch_on
        model.z = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1))  # switch_off
        model.p = pyo.Var(model.G_SET, model.T_SET)  # power
        model.r = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # spinning reserve
        model.c = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # piecewise linear cost
        model.l = pyo.Var(model.L_SET, model.T_SET)
    else:
        print("开始热启动")
        model.u = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1), initialize=u_start)  # is_on
        model.y = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1), initialize=y_start)  # switch_on
        model.z = pyo.Var(model.G_SET, model.T_SET, domain=pyo.Boolean, bounds=(0, 1), initialize=z_start)  # switch_off
        model.p = pyo.Var(model.G_SET, model.T_SET)  # power
        model.r = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # spinning reserve
        model.c = pyo.Var(model.G_SET, model.T_SET, bounds=(0, None))  # piecewise linear cost
        model.l = pyo.Var(model.L_SET, model.T_SET)
    # model.l = pyo.Var(model.L_SET, model.T_SET, bounds=lambda m, i, t: (-m.normal_flow_limit[i], m.normal_flow_limit[i]))  # line flow

    # fix variables
    for gi in model.G_SET:
        if model.must_run[gi]:
            model.u[gi, :].fix(1)
            model.y[gi, :].fix(0)
            model.z[gi, :].fix(0)
            if model.initial_status[gi] < 0:
                model.y[gi, 0].fix(1)
        if not model.provides_spinning_reserve[gi]:
            model.r[gi, :].fix(0)
    if use_storage:
        # 固定储能装置的初始储能量
        for i in range(sto_num):
            model.stopower[i, 0].fix(sto_initial_power[i])

    # constraints
    def min_uptime_rule(m, i, t):
        if model.must_run[i]:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.y[i, tau] for tau in range(max(t - m.min_uptime[i] + 1, 0), t + 1)) <= m.u[i, t]
    model.min_uptime_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=min_uptime_rule)

    def min_downtime_rule(m, i, t):
        if model.must_run[i]:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.z[i, tau] for tau in range(max(t - m.min_downtime[i] + 1, 0), t + 1)) <= 1 - m.u[i, t]
    model.min_downtime_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=min_downtime_rule)

    def init_min_time_rule(m, i):
        if model.must_run[i]:
            return pyo.Constraint.Skip
        if m.initial_status[i] > 0:
            if m.min_uptime[i] > m.initial_status[i]:
                return pyo.quicksum(m.z[i, t] for t in range(0, min(m.min_uptime[i] - m.initial_status[i], m.NT+1))) == 0
            else:
                return pyo.Constraint.Skip
        else:
            if m.min_downtime[i] > -m.initial_status[i]:
                return pyo.quicksum(m.y[i, t] for t in range(0, min(m.min_downtime[i] + m.initial_status[i], m.NT+1))) == 0
            else:
                return pyo.Constraint.Skip
    model.init_min_time_constraint = pyo.Constraint(model.G_SET, rule=init_min_time_rule)

    def logic_rule(m, i, t):
        if model.must_run[i]:
            return pyo.Constraint.Skip
        if t == 0:
            return m.u[i, t] - (1 if m.initial_status[i] > 0 else 0) == m.y[i, t] - m.z[i, t]
        else:
            return m.u[i, t] - m.u[i, t - 1] == m.y[i, t] - m.z[i, t]
    model.logic_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=logic_rule)
    model.logic_switch_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=lambda m, i, t: (m.y[i, t] + m.z[i, t]) <= 1)
    model.min_power_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=lambda m, i, t: m.p[i, t] >= m.u[i, t] * m.min_power[i, t])
    model.max_power_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=lambda m, i, t: m.p[i, t] + m.r[i, t] <= m.u[i, t] * m.max_power[i, t])

    def ramp_up_rule(m, i, t):
        if m.ramp_up_limit[i] >= 1e6 and m.startup_limit[i] >= 1e6:
            return pyo.Constraint.Skip
        actual_startup_limit = min(m.startup_limit[i], m.ramp_up_limit[i] + m.min_power[i, t])
        if t == 0:
            return m.p[i, t] - m.initial_power[i] + m.r[i, t] <= m.ramp_up_limit[i] * (1 if m.initial_status[i] > 0 else 0) + actual_startup_limit * m.y[i, t]
        else:
            return m.p[i, t] - m.p[i, t-1] + m.r[i, t] <= m.ramp_up_limit[i] * m.u[i, t-1] + actual_startup_limit * m.y[i, t]
    model.ramp_up_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=ramp_up_rule)

    def ramp_down_rule(m, i, t):
        if m.ramp_down_limit[i] >= 1e6 and m.shutdown_limit[i] >= 1e6:
            return pyo.Constraint.Skip
        actual_shutdown_limit = min(m.shutdown_limit[i], m.ramp_down_limit[i] + m.min_power[i, t])
        if t == 0:
            return m.initial_power[i] - m.p[i, t] <= m.ramp_down_limit[i] * m.u[i, t] + actual_shutdown_limit * m.z[i, t]
        else:
            return m.p[i, t-1] - m.p[i, t] <= m.ramp_down_limit[i] * m.u[i, t] + actual_shutdown_limit * m.z[i, t]
    model.ramp_down_constraint = pyo.Constraint(model.G_SET, model.T_SET, rule=ramp_down_rule)

    if use_storage:
        def line_flow_rule(m, i, t):
            if m.normal_flow_limit[i] >= 1e6 or len(G_TO_L_SET[i]) + len(B_TO_L_SET[i]) == 0:
                return pyo.Constraint.Skip
            return m.l[i, t] == pyo.quicksum(unit_to_line_factor[i][j] * m.p[j, t] for j in G_TO_L_SET[i]) - pyo.quicksum(bus_to_line_factor[i][j] * m.load_demand[j, t] for j in B_TO_L_SET[i]) + pyo.quicksum(ptdf[i][int(sto_storage_position[j])] * (m.pdis[j, t] - m.pch[j, t]) for j in range(sto_num))
            # return m.l[i, t] == pyo.quicksum(m.unit_to_line_factor[i][j] * m.p[j, t] for j in m.G_TO_L_SET[i]) - pyo.quicksum(m.bus_to_line_factor[i][j] * m.load_demand[j, t] for j in m.B_TO_L_SET[i])
    else:
        def line_flow_rule(m, i, t):
            if m.normal_flow_limit[i] >= 1e6 or len(G_TO_L_SET[i]) + len(B_TO_L_SET[i]) == 0:
                return pyo.Constraint.Skip
            return m.l[i, t] == pyo.quicksum(unit_to_line_factor[i][j] * m.p[j, t] for j in G_TO_L_SET[i]) - pyo.quicksum(bus_to_line_factor[i][j] * m.load_demand[j, t] for j in B_TO_L_SET[i])
            # return m.l[i, t] == pyo.quicksum(m.unit_to_line_factor[i][j] * m.p[j, t] for j in m.G_TO_L_SET[i]) - pyo.quicksum(m.bus_to_line_factor[i][j] * m.load_demand[j, t] for j in m.B_TO_L_SET[i])
    model.line_flow_constraint = pyo.Constraint(model.L_SET, model.T_SET, rule=line_flow_rule) # 这一句非常慢

    if use_storage:
        model.power_balance_constraint = pyo.Constraint(model.T_SET, rule=lambda m, t: pyo.quicksum(
            m.p[i, t] for i in m.G_SET) + pyo.quicksum(
            (m.pdis[i, t]-m.pch[i, t]) for i in range(sto_num)) == pyo.quicksum(m.load_demand[i, t] for i in m.B_SET))
    else:
        model.power_balance_constraint = pyo.Constraint(model.T_SET, rule=lambda m, t: pyo.quicksum(m.p[i, t] for i in m.G_SET) == pyo.quicksum(m.load_demand[i, t] for i in m.B_SET))
    model.reserve_constraint = pyo.Constraint(model.T_SET, rule=lambda m, t: pyo.quicksum(m.r[i, t] for i in m.G_SET) >= m.reserve_requirement[t])

    model.TransportLimits_1 = pyo.Constraint(model.L_SET, model.T_SET,rule=lambda m, i, t: m.l[i, t] >= -m.normal_flow_limit[i])
    model.TransportLimits_2 = pyo.Constraint(model.L_SET, model.T_SET,rule=lambda m, i, t: m.l[i, t] <= m.normal_flow_limit[i])

    if use_storage:
        # 储能相关约束
        # 充放电上限约束
        model.charge_rate_max_constraint = pyo.Constraint(range(sto_num), model.T_SET, rule=lambda m, i, t: m.pch[i, t] <= sto_p_ch_max[i])
        model.discharge_rate_max_constraint = pyo.Constraint(range(sto_num), model.T_SET,
                                                          rule=lambda m, i, t: m.pdis[i, t] <= sto_p_dis_max[i])
        # 容量逻辑约束
        model.capacity_logic_rule = pyo.Constraint(range(sto_num), range(model.NT-1),
                                                   rule = lambda m, i, t: m.stopower[i, t+1] == m.stopower[i, t] + sto_Eta[i] * m.pch[i, t] - (1/sto_Eta[i]) * m.pdis[i, t])

        # 储能上下限
        model.storage_max_constraints = pyo.Constraint(range(sto_num), model.T_SET, rule=lambda m, i, t: m.stopower[i, t] <= sto_max_power[i])
        model.storage_min_constraints = pyo.Constraint(range(sto_num), model.T_SET,
                                                       rule=lambda m, i, t: m.stopower[i, t] >= sto_min_power[i])
        model.storage_cost_constraint = pyo.Constraint(range(sto_num), model.T_SET, rule=lambda m, i, t: m.sto_c[i, t] == sto_Kbat[i] * (m.pdis[i, t]/sto_Eta[i] + m.pch[i, t] * sto_Eta[i]))

    # model.TransportLimits_1.clear()
    # model.TransportLimits_1._constructed = False
    # model.TransportLimits_1.construct()
    # model.TransportLimits_2.clear()
    # model.TransportLimits_2._constructed = False
    # model.TransportLimits_2.construct()

    # set objective
    n_seg = max(len(model.piecewise_cost[i, t]) for i in model.G_SET for t in model.T_SET)

    def piecewise_cost_obj_rule(m, i, t, i_seg):
        if i_seg + 1 < len(m.piecewise_cost[i, t]):
            if abs(m.piecewise_power[i, t][i_seg+1] - m.piecewise_power[i, t][i_seg]) < 1e-6:
                slope = 0
            else:
                slope = (m.piecewise_cost[i, t][i_seg+1] - m.piecewise_cost[i, t][i_seg]) / (m.piecewise_power[i, t][i_seg+1] - m.piecewise_power[i, t][i_seg])
            return m.c[i, t] - m.u[i, t] * m.piecewise_cost[i, t][i_seg] >= (m.p[i, t] - m.u[i, t] * m.piecewise_power[i, t][i_seg]) * slope
        elif i_seg == 0 and len(m.piecewise_cost[i, t]) == 1:
            # units not adjustable
            return m.c[i, t] - m.u[i, t] * m.piecewise_cost[i, t][i_seg] >= 0
        else:
            return pyo.Constraint.Skip
    model.piecewise_cost_obj_constraint = pyo.Constraint(model.G_SET, model.T_SET, range(max(1, n_seg-1)), rule=piecewise_cost_obj_rule)
    if use_storage:
        model.obj = pyo.Objective(expr=pyo.quicksum(model.c[i, t] + model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET) +
                                  pyo.quicksum(model.sto_c[i, t] for i in range(sto_num) for t in model.T_SET))
    else:
        model.obj = pyo.Objective(expr=pyo.quicksum(
            model.c[i, t] + model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET))
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print('model time: ' + str(model_time))

    # solve

    opt = pyo.SolverFactory('cplex')
    #opt.set_instance(model)
    solve_start = time.time()
    result = opt.solve(model, options={'mipgap': mip_gap, 'TimeLimit': time_limit}, tee=False, report_timing=True, warmstart=True)  # options={'mipgap': 0.001}
    solve_end = time.time()
    solve_t = -solve_start + solve_end
    print(f"求解时间为{solve_t}")
    stopower_data = [[pyo.value(model.stopower[i, t]) for t in model.T_SET] for i in range(1)]
    # 绘图展示
    # u_data = [[pyo.value(model.u[i, t]) for t in model.T_SET] for i in model.G_SET]
    # plt.plot(np.transpose([model.T_SET]), np.transpose(u_data))
    # plt.show()

    if not is_warm_start:
        u_start = {(i, t): round(pyo.value(model.u[i, t])) for i in model.G_SET for t in model.T_SET}
        y_start = {(i, t): round(pyo.value(model.y[i, t])) for i in model.G_SET for t in model.T_SET}
        z_start = {(i, t): round(pyo.value(model.z[i, t])) for i in model.G_SET for t in model.T_SET}
        p_start = {(i, t): pyo.value(model.p[i, t]) for i in model.G_SET for t in model.T_SET}
        r_start = {(i, t): pyo.value(model.r[i, t]) for i in model.G_SET for t in model.T_SET}
        c_start = {(i, t): pyo.value(model.c[i, t]) for i in model.G_SET for t in model.T_SET}
        l_start = {(i, t): pyo.value(model.l[i, t]) for i in model.L_SET for t in model.T_SET}
        print(f"优化的结果为：{pyo.value(model.obj)}")
        print(f'有潮流越限的结果为：{True in [False in [abs(pyo.value(model.l[i, t])) - model.normal_flow_limit[i] < 1e-3 for i in model.L_SET] for t in model.T_SET]}')
        model.clear()
        return model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, model_time, solve_t
    else:
        model.clear()
        return model, result, model_time, solve_t


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

    # 获取储能数据
    storage_data_name = 'storage_data.mat'
    data_dict_name = 'training_data'
    storage_data = _load_array(os.path.join(data_dict_name, 'storage_data.mat'))

    # 获取太阳能数据（未使用）
    solar_data = _load_array(os.path.join(data_dict_name, 'solar_gen'))
    solar_curve = solar_data['load'][0]

    # 获取测试数据
    case = "matpower/case89pegase/2017-07-13"
    file = gzip.open(path + case + '.json.gz')
    data = json.loads(file.read())

    # 获取模型数据和训练数据
    model_name_list = ['transformer_all_training_data_89pegase_data_not_random_1000_30000']
    model_name = model_name_list[0]
    dict_name = 'module_save'
    model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(model_name, dict_name)
    data_path = data_dict_name + '\\' + data_name
    inputs, targets, test_inputs, test_targets, time_num, input_size, output_size, average_load = get_data(model_type, data_path,
                                                                                             test_num,
                                                                                             training_num,
                                                                                             data_is_random=data_is_random,
                                                                                             cutted_by_time=cutted_by_time)
    # 下一步读取数据，然后测试热启动效果(全部热启动和只热启动确定的那一部分，最好还是专门写一个文件来执行)

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
        run_uc(data, is_warm_start=False, use_new_input=True, use_storage=True, use_new_min_up_down_time=True,
               use_solar= True, storage_data=storage_data, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen)
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
    model, result, model_time, solve_t = run_uc(data, is_warm_start=True, use_new_input=True, use_storage=True, use_new_min_up_down_time=True, storage_data=storage_data
                                                ,use_solar= True, u_start = u_start, y_start = y_start, z_start = z_start, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen)
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
