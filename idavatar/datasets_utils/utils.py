import numpy as np
import os
import yaml

class AllocateDatasets():
    def __init__(self, num_gpus, output_dir, dataset_dir):
        self.num_gpus = num_gpus
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        
    def allocate_datas(self):
        total_data_list = os.listdir(self.dataset_dir)
        num_gpus = self.num_gpus
        total_data_length = len(total_data_list)
        data_num_per_gpu = total_data_length // num_gpus
        file_index = 0

        for i in range(0, total_data_length, data_num_per_gpu):
            file_sublist = total_data_list[i:i+data_num_per_gpu]
            f = open('{}/gpu{}.txt'.format(self.output_dir, file_index), 'w')
            for j in file_sublist:
                f.write(str(j) + '\n')
            f.close
            file_index += 1

def load_config(config_file):
    with open('configs/datasets/detect.yaml') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result
        
