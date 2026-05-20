import numpy as np

from uc_gurobi_constraint_filter import run_constraint_filter
import numpy

def test_constraints_enable(model, device, test_inputs, test_targets, average_load, data, is_warm_start=False, u_start=0, y_start=0, z_start=0,
                            line_limit_rate=1, use_storage=False, storage_data=0, use_new_min_up_down_time = False, use_solar = False ,solar_gen = 0, new_add_num=5):
    model.eval()  # 改为测试模式

    device_cpu = 'cpu'
    device_cuda = 'cuda'

    # 使用模型获得线路启停预测结果
    bus_names, buses = zip(*data['Buses'].items())
    bus_num = len(buses)

    test_inputs = test_inputs.to(device)
    output = model(test_inputs)
    output = output.to(device_cpu)
    output = output.detach().numpy()
    data_num, output_size = output.shape

    test_inputs = test_inputs.to(device_cpu)
    test_inputs = test_inputs.detach().numpy()
    n_dim = test_targets.dim()
    if n_dim == 3:
        test_inputs = numpy.transpose(test_inputs, (0, 2, 1))
    else:
        time_num = int(len(test_inputs[0])/bus_num)
        test_inputs = numpy.reshape(test_inputs, [data_num, time_num, bus_num])
        test_inputs = numpy.transpose(test_inputs, axes=(0, 2, 1))

    all_one_time_num = 0
    all_round_num = []

    for data_i in range(data_num):
        # 可以考虑当有超过一般的时间某线路需要约束，则将这个线路的全时间约束加上
        constraints_enable = output[data_i]
        new_input = test_inputs[data_i]*average_load

        model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_model_time, all_solve_t, all_load_data_t, round_num, no_answer = \
            run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
                              use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=True,
                              use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input, storage_data=storage_data,
                              line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num, initial_constraints=constraints_enable)

        if round_num < 2:
            all_one_time_num = all_one_time_num + 1
        all_round_num.append(round_num)
        forecast_all_model_time = all_model_time
        forecast_all_solve_time = all_solve_t
        forecast_round_num = round_num
        print(f"使用预测结果时，建模总计用时{forecast_all_model_time}, 求解总计用时{forecast_all_solve_time}，循环次数:{forecast_round_num}")

        # enable_constraints = np.zeros_like(enable_constraints)
        # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_model_time, all_solve_t, round_num = \
        #     run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=True,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input, storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num, initial_constraints=enable_constraints)
        #
        # zero_all_model_time = all_model_time
        # zero_all_solve_time = all_solve_t
        # zero_round_num = round_num
        # print(f"从0开始启动，建模总计用时{zero_all_model_time}, 求解总计用时{zero_all_solve_time}，循环次数:{zero_round_num}")
        #
        # enable_constraints = np.ones_like(enable_constraints)
        # model, result, u_start, y_start, z_start, p_start, r_start, c_start, l_start, enable_constraints, all_model_time, all_solve_t, round_num = \
        #     run_constraint_filter(data, is_warm_start=is_warm_start, use_new_input=True, use_storage=use_storage,
        #                       use_new_min_up_down_time=use_new_min_up_down_time, use_initial_enable=True,
        #                       use_solar=use_solar, u_start=u_start, y_start=y_start, z_start=z_start, new_input=new_input, storage_data=storage_data,
        #                       line_limit_rate=line_limit_rate, solar_gen=solar_gen, new_add_num=new_add_num, initial_constraints=enable_constraints)
        #
        # one_all_model_time = all_model_time
        # one_all_solve_time = all_solve_t
        # one_round_num = round_num
        # print(f"从1开始启动，建模总计用时{one_all_model_time}, 求解总计用时{one_all_solve_time}，循环次数{one_round_num}")
        # print(f"从0开始启动，建模总计用时{zero_all_model_time}, 求解总计用时{zero_all_solve_time}，循环次数:{zero_round_num}")
        # print(f"使用预测结果时，建模总计用时{forecast_all_model_time}, 求解总计用时{forecast_all_solve_time}，循环次数:{forecast_round_num}")

    print(f"在{data_num}个数据中，有{all_one_time_num}个数据只循环一次，占比{100*all_one_time_num/data_num}%")
    return_data = {'all_one_time_num':all_one_time_num, 'all_roung_num':all_round_num}

    return return_data