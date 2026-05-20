import os
import time
import gzip
import json
import torch
import numpy as np
from unit_commitment_with_line_limit_save_cplex import run_uc, magnify_peak_valley_difference, get_load, get_solar_gen
from ml_basic_functions import get_data, ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


def _load_array(path):
    npz = path.replace('.mat', '.npz')
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io
    return scipy.io.loadmat(path)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

train_data_dict = 'training_data'

# 获取储能数据
storage_data = _load_array(os.path.join(train_data_dict, 'storage_data_1888rte.mat'))

# 获取太阳能数据（未使用）
solar_data = _load_array(train_data_dict + '\\' + 'solar_gen')
solar_curve = solar_data['load'][0]

# 获取测试数据
case = "matpower/case1888rte/2017-01-01"
file = gzip.open(path + case + '.json.gz')
data = json.loads(file.read())

# 获取gpu状态
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")


# 下一步读取数据，然后测试热启动效果(全部热启动和只热启动确定的那一部分，最好还是专门写一个文件来执行)
# 获取优化输入负载
test_i_index = 50

vanilla_all_model_time_sum = 0
vanilla_all_solve_t_sum = 0
vanilla_load_data_t_sum = 0
vanilla_round_num_sum = 0
vanilla_gap_sum = 0

no_initial_all_model_time_sum = 0
no_initial_all_solve_t_sum = 0
no_initial_load_data_t_sum = 0
no_initial_round_num_sum = 0

only_enable_all_model_time_sum = 0
only_enable_all_solve_t_sum = 0
only_enable_load_data_t_sum = 0
only_enable_round_num_sum = 0

all_true_all_model_time_sum = 0
all_true_all_solve_t_sum = 0
all_true_load_data_t_sum = 0
all_true_round_num_sum = 0

all_forecast_all_model_time_sum = 0
all_forecast_all_solve_t_sum = 0
all_forecast_load_data_t_sum = 0
all_forecast_round_num_sum = 0

all_reliable_all_model_time_sum = 0
all_reliable_all_solve_t_sum = 0
all_reliable_load_data_t_sum = 0
all_reliable_round_num_sum = 0

all_fix_all_model_time_sum = 0
all_fix_all_solve_t_sum = 0
all_fix_load_data_t_sum = 0
all_fix_round_num_sum = 0
all_fix_gap_sum = 0

fix_reliable_all_model_time_sum = 0
fix_reliable_all_solve_t_sum = 0
fix_reliable_load_data_t_sum = 0
fix_reliable_round_num_sum = 0
fix_reliable_gap_sum = 0

N = 0

for test_i_index in range(50, 60):

    # 获取模型数据和训练数据
    model_name_list = ['units_transformer_all_training_data_1888rte_data_is_random_1000_30000_256_0.0002',
                       'units_transformer_all_training_data_case118_storage_data_case118_30%.mat_data_is_random_1000_3500_64_0.0002','units_transformer_all_training_data_case300_storage_data_case300_60%.mat_data_is_random_1000_3000_64_0.0002','units_transformer_all_training_data_case1354pegase_data_is_random_1000_10000_256']
    model_name = model_name_list[0]
    dict_name = 'module_save'
    model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(
        model_name, dict_name)
    data_path = train_data_dict + '\\' + data_name
    inputs, targets, test_inputs, test_targets, time_num, input_size, output_size, average_load = get_data(model_type,
                                                                                                           data_path,
                                                                                                           test_num,
                                                                                                           training_num,
                                                                                                           data_is_random=data_is_random,
                                                                                                           cutted_by_time=cutted_by_time,
                                                                                                           random_seed=114514)

    n_dim = test_inputs.dim()

    if n_dim == 3:
        new_load_data = test_inputs[test_i_index, :, :] * average_load
        new_load_data = new_load_data.detach().numpy()
        new_load_data = np.transpose(new_load_data, axes=(1, 0))
    else:
        new_load_data = test_inputs[test_i_index*time_num:test_i_index*time_num+time_num, :] * average_load
        new_load_data = new_load_data.detach().numpy()
        new_load_data = np.transpose(new_load_data, axes=(1, 0))
        # new_load_data = test_inputs[test_i_index, :] * average_load
        # new_load_data = new_load_data.detach().numpy()
        # all_size = len(new_load_data)
        # output_size = int(all_size/time_num)
        # new_load_data = np.reshape(new_load_data, [time_num, output_size])
        # new_load_data = np.transpose(new_load_data, axes=(1, 0))

    # 获取预测热启动
    low_gate = 0.1
    high_gate = 0.9
    if n_dim == 3:
        test_input_i = test_inputs[test_i_index, :, :]
    else:
        test_input_i = test_inputs[test_i_index*time_num:test_i_index*time_num+time_num, :]
    test_input_i = test_input_i.unsqueeze(0)
    test_input_i = test_input_i.to(device)

    model = model.to(device)
    forecast_output = model(test_input_i)
    forecast_output = forecast_output.to(device_cpu)
    forecast_output = forecast_output.detach().numpy()
    is_reliable_matrix = (forecast_output <= low_gate) + (forecast_output >= high_gate)
    # if n_dim == 3:
    data_num, n_t, output_size = forecast_output.shape
    # else:
    #     data_num, all_size = forecast_output.shape


    reliable_warm_start_u = {}
    reliable_warm_start_y = {}
    reliable_warm_start_z = {}
    all_warm_start_u = {}
    all_warm_start_y = {}
    all_warm_start_z = {}

    # 获取u的热启动值
    # if n_dim == 3:
    for j in range(output_size):
        for t in range(time_num):
            is_reliable = is_reliable_matrix[0, t, j]
            all_warm_start_u[j, t] = round(forecast_output[0, t, j])
            if is_reliable:
                reliable_warm_start_u[j, t] = round(forecast_output[0, t, j])

    # 获取y, z的热启动值
    for j in range(output_size):
        for t in range(1, time_num):
            is_reliable_t = is_reliable_matrix[0, t, j]
            is_reliable_t_1 = is_reliable_matrix[0, t-1, j]
            delta = all_warm_start_u[j, t] - all_warm_start_u[j, t-1]
            if delta > 0.5:
                all_warm_start_y[j, t] = int(1)
                all_warm_start_z[j, t] = int(0)
            elif delta < -0.5:
                all_warm_start_y[j, t] = int(0)
                all_warm_start_z[j, t] = int(1)
            else:
                all_warm_start_y[j, t] = int(0)
                all_warm_start_z[j, t] = int(0)
            if is_reliable_t and is_reliable_t_1:
                delta = reliable_warm_start_u[j, t] - reliable_warm_start_u[j, t - 1]
                if delta > 0.5:
                    reliable_warm_start_y[j, t] = int(1)
                    reliable_warm_start_z[j, t] = int(0)
                elif delta < -0.5:
                    reliable_warm_start_y[j, t] = int(0)
                    reliable_warm_start_z[j, t] = int(1)
                else:
                    reliable_warm_start_y[j, t] = int(0)
                    reliable_warm_start_z[j, t] = int(0)
    # elif n_dim == 2:
    #     for j in range(output_size):
    #         for t in range(time_num):
    #             is_reliable = is_reliable_matrix[0, t*output_size+j]
    #             all_warm_start_u[j, t] = round(forecast_output[0, t*output_size+j])
    #             if is_reliable:
    #                 reliable_warm_start_u[j, t] = round(forecast_output[0, t*output_size+j])
    #
    #     # 获取y, z的热启动值
    #     for j in range(output_size):
    #         for t in range(1, time_num):
    #             is_reliable_t = is_reliable_matrix[0, t*output_size+j]
    #             is_reliable_t_1 = is_reliable_matrix[0, (t-1)*output_size+j]
    #             delta = all_warm_start_u[j, t] - all_warm_start_u[j, t-1]
    #             if delta > 0.5:
    #                 all_warm_start_y[j, t] = int(1)
    #                 all_warm_start_z[j, t] = int(0)
    #             elif delta < -0.5:
    #                 all_warm_start_y[j, t] = int(0)
    #                 all_warm_start_z[j, t] = int(1)
    #             else:
    #                 all_warm_start_y[j, t] = int(0)
    #                 all_warm_start_z[j, t] = int(0)
    #             if is_reliable_t and is_reliable_t_1:
    #                 delta = reliable_warm_start_u[j, t] - reliable_warm_start_u[j, t - 1]
    #                 if delta > 0.5:
    #                     reliable_warm_start_y[j, t] = int(1)
    #                     reliable_warm_start_z[j, t] = int(0)
    #                 elif delta < -0.5:
    #                     reliable_warm_start_y[j, t] = int(0)
    #                     reliable_warm_start_z[j, t] = int(1)
    #                 else:
    #                     reliable_warm_start_y[j, t] = int(0)
    #                     reliable_warm_start_z[j, t] = int(0)

    # 预测起作用约束
    # 获取约束预测模型
    enable_model_name_list = ['constraints_DNN_all_training_data_1888rte_data_is_random_1000_10000_[256, 256, 256]', 'constraints_DNN_all_training_data_case118_storage_data_case118_30%.mat_data_is_random_1000_20000_[256, 256, 256]_0.0002','constraints_DNN_all_training_data_case300_storage_data_case300_30%.mat_data_is_random_1000_20000_[256, 256, 256]_0.0002','constraints_DNN_all_training_data_case1354pegase_data_is_random_1000_5000_[256, 256, 256]']
    model_name = enable_model_name_list[0]
    dict_name = 'module_save'
    enable_model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(
        model_name, dict_name)

    inputs, targets, test_inputs, test_targets, time_num, input_size, output_size, average_load = get_data(model_type, data_path,
                                                                                             test_num,
                                                                                             training_num,
                                                                                             data_is_random=data_is_random,
                                                                                             cutted_by_time=cutted_by_time)
    test_input_i = test_inputs[test_i_index, :]
    test_input_i = test_input_i.unsqueeze(0)
    test_input_i = test_input_i.to(device)
    enable_model = enable_model.to(device)

    forecast_enable = enable_model(test_input_i)
    forecast_enable = forecast_enable.to(device_cpu)
    forecast_enable = forecast_enable.squeeze(0)
    forecast_enable = forecast_enable.detach().numpy()
    # forecast_enable = numpy.transpose(forecast_enable, axes=(1, 0))

    # 增大峰谷差
    # magnify_rate = 1.05
    line_limit_rate = 1
    solar_base_power = 3000
    solar_gen = get_solar_gen(solar_curve, solar_base_power)

    # load_data = get_load(data)
    # new_load_data = magnify_peak_valley_difference(load_data, magnify_rate)

    test_warmstart = False
    if test_warmstart:
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

        # 开始热启动，此时热启动数值为全部值
        start = time.time()
        model, result, model_time, solve_t = run_uc(data, is_warm_start=True, use_new_input=True, use_storage=True, use_new_min_up_down_time=True, storage_data=storage_data
                                                    ,use_solar= True, u_start = all_warm_start_u, y_start = all_warm_start_y, z_start = all_warm_start_z, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen)
        #    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
        end = time.time()
        print(f"求解结果为：{results}")
        print(f"建模时间为：{model_time}, 求解时间为：{-model_time + end - start}, 总时间为：{end-start}, solve_t为{solve_t}")
        model_time_all_warmstart = model_time
        solve_time_all_warmstart = -model_time + end - start
        total_time_all_warmstart = end - start
        solve_t_all_warmstart = solve_t

        # 开始热启动，此时热启动数值为可靠值
        start = time.time()
        model, result, model_time, solve_t = run_uc(data, is_warm_start=True, use_new_input=True, use_storage=True, use_new_min_up_down_time=True, storage_data=storage_data
                                                    ,use_solar= True, u_start = reliable_warm_start_u, y_start = reliable_warm_start_y, z_start = reliable_warm_start_z, new_input=new_load_data, line_limit_rate = line_limit_rate, solar_gen=solar_gen)
        #    results[case] = (pyo.value(model.obj), pyo.value(pyo.quicksum(model.c[i, t] for i in model.G_SET for t in model.T_SET)), pyo.value(pyo.quicksum(model.startup_cost[i] * model.y[i, t] for i in model.G_SET for t in model.T_SET)))
        end = time.time()
        print(f"求解结果为：{results}")
        print(f"建模时间为：{model_time}, 求解时间为：{-model_time + end - start}, 总时间为：{end-start}, solve_t为{solve_t}")
        model_time_reliable_warmstart = model_time
        solve_time_reliable_warmstart = -model_time + end - start
        total_time_reliable_warmstart = end - start
        solve_t_reliable_warmstart = solve_t

        print(f"求解结果为：{results}")
        print(f"无热启动时，建模时间为：{model_time_no_warmstart}, 求解时间为：{solve_time_no_warmstart}, 总时间为：{total_time_no_warmstart}, solve_t为{solve_t_no_warmstart}")
        print(f"按照全部预测正确，热启动时，建模时间为：{model_time_all_right_warmstart}, 求解时间为：{solve_time_all_right_warmstart}, 总时间为：{total_time_all_right_warmstart}, solve_t为{solve_t_all_right_warmstart}")
        print(f"全部热启动，热启动时，建模时间为：{model_time_all_warmstart}, 求解时间为：{solve_time_all_warmstart}, 总时间为：{total_time_all_warmstart}, solve_t为{solve_t_all_warmstart}")
        print(f"只热启动可靠值，热启动时，建模时间为：{model_time_reliable_warmstart}, 求解时间为：{solve_time_reliable_warmstart}, 总时间为：{total_time_reliable_warmstart}, solve_t为{solve_t_reliable_warmstart}")
    else:

        # 无热启动，全部约束启动
        is_warm_start = False
        use_storage = True
        use_new_min_up_down_time = True
        use_initial_enable = True
        use_solar = True
        u_start = reliable_warm_start_u
        y_start = reliable_warm_start_y
        z_start = reliable_warm_start_z
        new_input = new_load_data
        new_add_num = 5
        time_limit = 1e6
        # constraints_enable = np.ones_like(forecast_enable)
        # model, vanilla_result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, vanilla_all_model_time, vanilla_all_solve_t, vanilla_load_data_t, vanilla_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable, time_limit=time_limit)
        # print(f"无热启动，全部约束启动时，全部建模时间为{vanilla_all_model_time}，全部求解时间为{vanilla_all_solve_t}，全部读取数据时间为{vanilla_load_data_t},循环次数为{vanilla_round_num}")

        # # 无热启动,无初始起作用约束
        # is_warm_start = False
        # use_initial_enable = False
        # constraints_enable = np.ones_like(forecast_enable)
        # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, no_initial_all_model_time, no_initial_all_solve_t, no_initial_load_data_t, no_initial_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable)
        # print(
        #     f"无热启动，无初始起作用约束，全部建模时间为{no_initial_all_model_time}，全部求解时间为{no_initial_all_solve_t}，全部读取数据时间为{no_initial_load_data_t},循环次数为{no_initial_round_num}")
        #
        # # 无热启动，预测约束启动
        # is_warm_start = False
        # use_initial_enable = True
        # constraints_enable = forecast_enable
        # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints1, only_enable_all_model_time, only_enable_all_solve_t, only_enable_load_data_t, only_enable_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable)
        # print(
        #     f"无热启动，预测约束启动，全部建模时间为{only_enable_all_model_time}，全部求解时间为{only_enable_all_solve_t}，全部读取数据时间为{only_enable_load_data_t},循环次数为{only_enable_round_num}")
        #
        # # 热启动全部正确值，预测约束启动
        # is_warm_start = True
        # use_initial_enable = True
        # constraints_enable = forecast_enable
        # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_true_all_model_time, all_true_all_solve_t,all_true_load_data_t, all_true_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable)
        # print(
        #     f"热启动全部正确值，预测约束启动，全部建模时间为{all_true_all_model_time}，全部求解时间为{all_true_all_solve_t}，全部读取数据时间为{all_true_load_data_t},循环次数为{all_true_round_num}")

        # 热启动全部预测值，预测约束启动
        is_warm_start = True
        use_initial_enable = True
        constraints_enable = forecast_enable
        u_start = all_warm_start_u
        y_start = all_warm_start_y
        z_start = all_warm_start_z
        model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_forecast_all_model_time, all_forecast_all_solve_t,all_forecast_load_data_t, all_forecast_round_num, no_answer = \
        run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
                              use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
                              use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
                              storage_data=storage_data,
                              line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
                              initial_constraints=constraints_enable)
        print(
            f"热启动全部预测值，预测约束启动，全部建模时间为{all_forecast_all_model_time}，全部求解时间为{all_forecast_all_solve_t}，全部读取数据时间为{all_forecast_load_data_t},循环次数为{all_forecast_round_num}")

        # 热启动全部可靠值，预测约束启动
        is_warm_start = True
        use_initial_enable = True
        constraints_enable = forecast_enable
        u_start = reliable_warm_start_u
        y_start = reliable_warm_start_y
        z_start = reliable_warm_start_z
        model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_reliable_all_model_time, all_reliable_all_solve_t, all_reliable_load_data_t, all_reliable_round_num, no_answer = \
        run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
                              use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
                              use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
                              storage_data=storage_data,
                              line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
                              initial_constraints=constraints_enable)
        print(
            f"热启动全部可靠值，预测约束启动，全部建模时间为{all_reliable_all_model_time}，全部求解时间为{all_reliable_all_solve_t}，全部读取数据时间为{all_reliable_load_data_t},循环次数为{all_reliable_round_num}")
        vanilla_gap = 100*(vanilla_result - result)/result

        # # 固定全部值，预测约束启动
        # is_warm_start = False
        # use_initial_enable = True
        # fix_int = True
        # constraints_enable = forecast_enable
        # u_start = all_warm_start_u
        # y_start = all_warm_start_y
        # z_start = all_warm_start_z
        # model, all_fix_result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_fix_all_model_time, all_fix_all_solve_t, all_fix_load_data_t, all_fix_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable, fix_int=fix_int)
        # all_fix_gap = 100 * (all_fix_result - result)/result
        # print(
        #     f"固定全部值，预测约束启动，全部建模时间为{all_fix_all_model_time}，全部求解时间为{all_fix_all_solve_t}，全部读取数据时间为{all_fix_load_data_t},循环次数为{all_fix_round_num},gap为{all_fix_gap}%")

        # # 固定全部可靠值，预测约束启动
        # is_warm_start = False
        # use_initial_enable = True
        # constraints_enable = forecast_enable
        # fix_int = True
        # u_start = reliable_warm_start_u
        # y_start = reliable_warm_start_y
        # z_start = reliable_warm_start_z
        # model, fix_reliable_result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, fix_reliable_all_model_time, fix_reliable_all_solve_t, fix_reliable_load_data_t, fix_reliable_round_num, no_answer = \
        # run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=use_initial_enable,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input,
        #                       storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num,
        #                       initial_constraints=constraints_enable, fix_int=fix_int)
        # fix_reliable_gap = 100 * (fix_reliable_result - result)/result
        # print(
        #     f"固定全部可靠值，预测约束启动，全部建模时间为{fix_reliable_all_model_time}，全部求解时间为{fix_reliable_all_solve_t}，全部读取数据时间为{fix_reliable_load_data_t},循环次数为{fix_reliable_round_num},gap为{fix_reliable_gap}%")

        print('\n给出全部对比：\n')
        print(
            f"无热启动，全部约束启动时，全部建模时间为{vanilla_all_model_time}，全部求解时间为{vanilla_all_solve_t}，全部读取数据时间为{vanilla_load_data_t},循环次数为{vanilla_round_num}")
        # print(
        #     f"无热启动，无初始约束，全部建模时间为{no_initial_all_model_time}，全部求解时间为{no_initial_all_solve_t}，全部读取数据时间为{no_initial_load_data_t},循环次数为{no_initial_round_num}")
        # print(
        #     f"无热启动，预测约束启动，全部建模时间为{only_enable_all_model_time}，全部求解时间为{only_enable_all_solve_t}，全部读取数据时间为{only_enable_load_data_t},循环次数为{only_enable_round_num}")
        # print(
        #     f"热启动全部正确值，预测约束启动，全部建模时间为{all_true_all_model_time}，全部求解时间为{all_true_all_solve_t}，全部读取数据时间为{all_true_load_data_t},循环次数为{all_true_round_num}")
        print(
            f"热启动全部预测值，预测约束启动，全部建模时间为{all_forecast_all_model_time}，全部求解时间为{all_forecast_all_solve_t}，全部读取数据时间为{all_forecast_load_data_t},循环次数为{all_forecast_round_num}")
        print(
            f"热启动全部可靠值，预测约束启动，全部建模时间为{all_reliable_all_model_time}，全部求解时间为{all_reliable_all_solve_t}，全部读取数据时间为{all_reliable_load_data_t},循环次数为{all_reliable_round_num}")
        # print(
        #     f"固定全部值，预测约束启动，全部建模时间为{all_fix_all_model_time}，全部求解时间为{all_fix_all_solve_t}，全部读取数据时间为{all_fix_load_data_t},循环次数为{all_fix_round_num},gap为{all_fix_gap}%")
        # print(
        #     f"固定全部可靠值，预测约束启动，全部建模时间为{fix_reliable_all_model_time}，全部求解时间为{fix_reliable_all_solve_t}，全部读取数据时间为{fix_reliable_load_data_t},循环次数为{fix_reliable_round_num},gap为{fix_reliable_gap}%")


        vanilla_all_model_time_sum = vanilla_all_model_time_sum + vanilla_all_model_time
        vanilla_all_solve_t_sum = vanilla_all_solve_t_sum + vanilla_all_solve_t
        vanilla_load_data_t_sum = vanilla_load_data_t_sum + vanilla_load_data_t
        vanilla_round_num_sum = vanilla_round_num_sum + vanilla_round_num
        if vanilla_gap < 0:
            vanilla_gap = 0
        vanilla_gap_sum = vanilla_gap_sum + vanilla_gap

        # no_initial_all_model_time_sum = no_initial_all_model_time_sum + no_initial_all_model_time
        # no_initial_all_solve_t_sum = no_initial_all_solve_t_sum + no_initial_all_solve_t
        # no_initial_load_data_t_sum = no_initial_load_data_t_sum + no_initial_load_data_t
        # no_initial_round_num_sum = no_initial_round_num_sum + no_initial_round_num

        # only_enable_all_model_time_sum = only_enable_all_model_time_sum + only_enable_all_model_time
        # only_enable_all_solve_t_sum = only_enable_all_solve_t_sum + only_enable_all_solve_t
        # only_enable_load_data_t_sum = only_enable_load_data_t_sum + only_enable_load_data_t
        # only_enable_round_num_sum = only_enable_round_num_sum + only_enable_round_num

        # all_true_all_model_time_sum = all_true_all_model_time_sum + all_true_all_model_time
        # all_true_all_solve_t_sum = all_true_all_solve_t_sum + all_true_all_solve_t
        # all_true_load_data_t_sum = all_true_load_data_t_sum + all_true_load_data_t
        # all_true_round_num_sum = all_true_round_num_sum + all_true_round_num

        all_forecast_all_model_time_sum = all_forecast_all_model_time_sum + all_forecast_all_model_time
        all_forecast_all_solve_t_sum = all_forecast_all_solve_t_sum + all_forecast_all_solve_t
        all_forecast_load_data_t_sum = all_forecast_load_data_t_sum + all_forecast_load_data_t
        all_forecast_round_num_sum = all_forecast_round_num_sum + all_forecast_round_num

        all_reliable_all_model_time_sum = all_reliable_all_model_time_sum + all_reliable_all_model_time
        all_reliable_all_solve_t_sum = all_reliable_all_solve_t_sum + all_reliable_all_solve_t
        all_reliable_load_data_t_sum = all_reliable_load_data_t_sum + all_reliable_load_data_t
        all_reliable_round_num_sum = all_reliable_round_num_sum + all_reliable_round_num

        # all_fix_all_model_time_sum = all_fix_all_model_time_sum + all_fix_all_model_time
        # all_fix_all_solve_t_sum = all_fix_all_solve_t_sum + all_fix_all_solve_t
        # all_fix_load_data_t_sum = all_fix_load_data_t_sum + all_fix_load_data_t
        # all_fix_round_num_sum = all_fix_round_num_sum + all_fix_round_num
        # all_fix_gap_sum = all_fix_gap_sum + all_fix_gap

        # fix_reliable_all_model_time_sum = fix_reliable_all_model_time_sum + fix_reliable_all_model_time
        # fix_reliable_all_solve_t_sum = fix_reliable_all_solve_t_sum + fix_reliable_all_solve_t
        # fix_reliable_load_data_t_sum = fix_reliable_load_data_t_sum + fix_reliable_load_data_t
        # fix_reliable_round_num_sum = fix_reliable_round_num_sum + fix_reliable_round_num
        # if fix_reliable_gap < 0:
        #     fix_reliable_gap = 0
        # fix_reliable_gap_sum = fix_reliable_gap_sum + fix_reliable_gap

        N = N+1

print('\n给出全部对比：\n')
print(
    f"无热启动，全部约束启动时，全部建模时间为{vanilla_all_model_time_sum/N}，全部求解时间为{vanilla_all_solve_t_sum/N}，全部读取数据时间为{vanilla_load_data_t_sum/N},循环次数为{vanilla_round_num_sum/N},gap为{vanilla_gap_sum/N}")
# print(
#     f"无热启动，无初始约束，全部建模时间为{no_initial_all_model_time_sum/N}，全部求解时间为{no_initial_all_solve_t_sum/N}，全部读取数据时间为{no_initial_load_data_t_sum/N},循环次数为{no_initial_round_num_sum/N}")
# print(
#     f"无热启动，预测约束启动，全部建模时间为{only_enable_all_model_time_sum/N}，全部求解时间为{only_enable_all_solve_t_sum/N}，全部读取数据时间为{only_enable_load_data_t_sum/N},循环次数为{only_enable_round_num_sum/N}")
# print(
#     f"热启动全部正确值，预测约束启动，全部建模时间为{all_true_all_model_time_sum/N}，全部求解时间为{all_true_all_solve_t_sum/N}，全部读取数据时间为{all_true_load_data_t_sum/N},循环次数为{all_true_round_num_sum/N}")
print(
    f"热启动全部预测值，预测约束启动，全部建模时间为{all_forecast_all_model_time_sum/N}，全部求解时间为{all_forecast_all_solve_t_sum/N}，全部读取数据时间为{all_forecast_load_data_t_sum/N},循环次数为{all_forecast_round_num_sum/N}")
print(
    f"热启动全部可靠值，预测约束启动，全部建模时间为{all_reliable_all_model_time_sum/N}，全部求解时间为{all_reliable_all_solve_t_sum/N}，全部读取数据时间为{all_reliable_load_data_t_sum/N},循环次数为{all_reliable_round_num_sum/N}")
# print(
#     f"固定全部值，预测约束启动，全部建模时间为{all_fix_all_model_time_sum/N}，全部求解时间为{all_fix_all_solve_t_sum/N}，全部读取数据时间为{all_fix_load_data_t_sum/N},循环次数为{all_fix_round_num_sum/N},gap为{all_fix_gap_sum/N}%")
# print(
#     f"固定全部可靠值，预测约束启动，全部建模时间为{fix_reliable_all_model_time_sum/N}，全部求解时间为{fix_reliable_all_solve_t_sum/N}，全部读取数据时间为{fix_reliable_load_data_t_sum/N},循环次数为{fix_reliable_round_num_sum/N},gap为{fix_reliable_gap_sum/N}%")

    # for k in u_start1.keys():
    #     if abs(u_start1[k]-u_start2[k])>0.5:
    #         print(f"在{k}处u不一样，1为{u_start1[k]}，2为{u_start2}")
    #     if abs(p_start1[k]-p_start2[k])>1e-3:
    #         print(f"在{k}处p不一样，1为{p_start1[k]}，2为{p_start2}")
    #
    # for i in range(210):
    #     for t in range(30):
    #         if abs(enable_constraints1[i]-enable_constraints2[i])>0.5:
    #             print(f"在{(i, t)}处的constraints不同，1为{enable_constraints1[i]}，2为{enable_constraints2[i]}")