import scipy.io
import numpy as np

def read_training_data(training_data):
    learning_data = scipy.io.loadmat(training_data)
    input_data = learning_data["input_data"]
    output_data = learning_data["output_data"]
    output_percentage = np.sum(output_data, axis=0)/len(output_data)
    return input_data, output_percentage

if __name__ == '__main__':
    dict_name = 'training_data'
    data_name = 'all_training_data_case1888rte_pro.mat'
    training_data = dict_name + '\\' + data_name

    input_data, output_percentage = read_training_data(training_data)

    dict_name = 'training_data'
    data_name = 'all_training_data_case1888rte_pro_newinput.mat'
    training_data = dict_name + '\\' + data_name

    input_data_newinput, output_percentage_newinput = read_training_data(training_data)

    output_percentage_delta = abs(output_percentage - output_percentage_newinput)
    input_delta = abs(input_data - input_data_newinput)

    c = 1

