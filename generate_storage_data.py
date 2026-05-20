import scipy.io
import numpy as np
import os

# 1.峰谷差改变 2.加储能 3.改变最小启停时间（未验证紧凑型）
# 价格贵一点，容量大一点，这样保证有解的情况下更多的调用机组
# 目前的问题是总能量不够

# max_power = [5000, 5000]
# min_power = [500, 500]
# initial_power = [1000, 1000]
# storage_position = [10, 50] #储能位于发电机的节点编号
# Kbat = [10, 15] # 充放电成本
# Eta = [0.93, 0.9] # 充放电损耗
# p_ch_max = [500, 700]
# p_dis_max = [500, 700]

# max_power = [5000]*12 # 1354pegase
# min_power = [500]*12
# initial_power = [1000]*12
# storage_position = [100*i for i in range(1, 13)] #储能位于发电机的节点编号
# Kbat = [12.5]*12 # 充放电成本
# Eta = [0.91]*12 # 充放电损耗
# p_ch_max = [500]*12
# p_dis_max = [500]*12

# max_power = [5000]*15 # 1354pegase
# min_power = [500]*15
# initial_power = [1000]*15
# storage_position = [100*i for i in range(1, 16)] #储能位于发电机的节点编号
# Kbat = [12.5]*15 # 充放电成本
# Eta = [0.91]*15 # 充放电损耗
# p_ch_max = [500]*15
# p_dis_max = [500]*15

# max_power = [2500]*15 # 1354pegase_100%
# min_power = [250]*15
# initial_power = [500]*15
# storage_position = [80*i for i in range(1, 16)] #储能位于发电机的节点编号
# Kbat = [12.5]*15 # 充放电成本
# Eta = [0.91]*15 # 充放电损耗
# p_ch_max = [250]*15
# p_dis_max = [250]*15

# max_power = [5000]*15 # case1888rte
# min_power = [500]*15
# initial_power = [1000]*15
# storage_position = [100*i for i in range(1, 16)] #储能位于发电机的节点编号
# Kbat = [12.5]*40 # 充放电成本
# Eta = [0.91]*40 # 充放电损耗
# p_ch_max = [500]*40
# p_dis_max = [500]*40

max_power = [10000]*15 # case1888rte_200%
min_power = [1000]*15
initial_power = [2000]*15
storage_position = [100*i for i in range(1, 16)] #储能位于发电机的节点编号
Kbat = [12.5]*40 # 充放电成本
Eta = [0.91]*40 # 充放电损耗
p_ch_max = [500]*40
p_dis_max = [500]*40

# max_power = [2500]*15 # case1888rte_50%
# min_power = [250]*15
# initial_power = [500]*15
# storage_position = [100*i for i in range(1, 16)] #储能位于发电机的节点编号
# Kbat = [12.5]*15 # 充放电成本
# Eta = [0.91]*15 # 充放电损耗
# p_ch_max = [250]*40
# p_dis_max = [250]*40

# max_power = [0]*15 # case1888rte_0%
# min_power = [0]*15
# initial_power = [0]*15
# storage_position = [100*i for i in range(1, 16)] #储能位于发电机的节点编号
# Kbat = [12.5]*40 # 充放电成本
# Eta = [0.91]*40 # 充放电损耗
# p_ch_max = [500]*40
# p_dis_max = [500]*40

# 价格贵一点，容量大一点，这样保证有解的情况下更多的调用机组
# max_power = [200000, 200000]
# min_power = [50, 50]
# initial_power = [40000, 40000]
# storage_position = [23, 70] #储能位于发电机的节点编号
# Kbat = [2.5, 3] # 充放电成本
# Eta = [0.93, 0.9] # 充放电损耗
# p_ch_max = [1000, 1500]
# p_dis_max = [1000, 1500]

# max_power = [2000]
# min_power = [50]
# initial_power = [400]
# storage_position = [10] #储能位于发电机的节点编号
# Kbat = [2.5] # 充放电成本
# Eta = [0.93] # 充放电损耗
# p_ch_max = [100]
# p_dis_max = [100]

# max_power = [15]*6 # case30_30%
# min_power = [0]*6
# initial_power = [0]*6
# storage_position = [5*i for i in range(0, 6)] #储能位于发电机的节点编号
# Kbat = [12.5]*15 # 充放电成本
# Eta = [0.98]*15 # 充放电损耗
# p_ch_max = [6]*6
# p_dis_max = [6]*6

# max_power = [200]*30 # case300_60%
# min_power = [0]*30
# initial_power = [0]*30
# storage_position = [10*i for i in range(0, 30)] #储能位于发电机的节点编号
# Kbat = [12.5]*30 # 充放电成本
# Eta = [0.98]*30 # 充放电损耗
# p_ch_max = [80]*30
# p_dis_max = [80]*30

# max_power = [600]*30 # case300_180%
# min_power = [0]*30
# initial_power = [0]*30
# storage_position = [10*i for i in range(0, 30)] #储能位于发电机的节点编号
# Kbat = [12.5]*30 # 充放电成本
# Eta = [0.98]*30 # 充放电损耗
# p_ch_max = [240]*30
# p_dis_max = [240]*30

# max_power = [30]*30 # case118_30%
# min_power = [0]*30
# initial_power = [0]*30
# storage_position = [3*i for i in range(0, 30)] #储能位于发电机的节点编号
# Kbat = [12.5]*30 # 充放电成本
# Eta = [0.98]*30 # 充放电损耗
# p_ch_max = [12]*30
# p_dis_max = [12]*30

dict_name = 'training_data'
data_name = 'storage_data_case1888rte_200%'
os.makedirs(dict_name, exist_ok=True)

mat_path = os.path.join(dict_name, data_name + '.mat')
npz_path = os.path.join(dict_name, data_name + '.npz')

save_dict = {'max_power': max_power, 'min_power': min_power, 'initial_power': initial_power,
             'storage_position': storage_position, 'Kbat': Kbat, 'Eta': Eta, 'p_ch_max': p_ch_max,
             'p_dis_max': p_dis_max}


# 保存为 .mat（兼容旧代码）
scipy.io.savemat(mat_path, save_dict)

# 保存为 .npz（新格式，优先加载）
np.savez_compressed(npz_path, **{k: np.array(v) for k, v in save_dict.items()})
