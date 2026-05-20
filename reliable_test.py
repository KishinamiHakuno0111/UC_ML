import os
import time
import gzip
import json
import torch

import numpy as np
import scipy.io
import numpy
from unit_commitment_with_line_limit_save_cplex import run_uc, magnify_peak_valley_difference, get_load, get_solar_gen
from ml_basic_functions import get_data, ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_dict = 'training_data'

# 获取gpu状态
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

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

# 获取预测热启动
low_gate = 0.1
high_gate = 0.9
if n_dim == 3:
    test_input_i = test_inputs
else:
    test_input_i = test_inputs
# test_input_i = test_input_i.unsqueeze(0)
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

forecast_output_array = np.reshape(forecast_output, (data_num* n_t * output_size))

plt.hist(forecast_output_array, bins=100, density=True)
plt.show()

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