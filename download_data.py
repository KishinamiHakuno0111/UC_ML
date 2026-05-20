import requests
import os

case_name = 'case118'
os.mkdir("case_data/matpower/" + case_name)
data_year = '2017'
head = 'https://axavier.org/UnitCommitment.jl/0.4/instances/matpower/' + case_name + '/'

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
        file_name = data_year + '-' + month_str + '-' + day_str + '.json.gz'
        url = head + file_name
        save_path = 'case_data/matpower/' + case_name + '/' + file_name  # 请替换为你想要的路径和文件名
        response = requests.get(url)
        # 检查请求是否成功
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"{file_name} has been downloaded")

print('文件下载完成')