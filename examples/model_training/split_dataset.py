from ase.io import read, write
from ase.db import connect
import math
import random
import torch
import numpy as np
from tqdm import tqdm
import os

def random_state(seed_value):
    # random seed setting to ensure that the experimental results can be reproduced
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

def split_mace_dataset(dataset_name, db_path, split_ratio):
    db = read(f'{db_path}/{dataset_name}.xyz', ':')
    num_data = len(db)
    print("Number of data:", num_data)
    # 随机打乱数据集
    random_state(42)
    random.shuffle(db)

    tag1 = math.ceil(num_data * split_ratio[0])

    # 划分数据集
    train_data = db[:tag1]
    train_data_idx = [i.info['mol_name'] for i in db[:tag1]]
    test_data = db[tag1:]
    test_data_idx = [i.info['mol_name'] for i in db[tag1:]]

    # 写入文件
    # non-periodic data is handled correctly by MACE, so we do not need to change anything
    write(f'{db_path}/{dataset_name}_train.xyz', train_data)
    write(f'{db_path}/{dataset_name}_test.xyz', test_data)

if __name__ == '__main__':

    random.seed(42) # set random seed for reproducibility
    dataset_name = 'demo'
    db_path = r'./'
    split_ratio = [0.9, 0.1]

    split_mace_dataset(dataset_name, db_path, split_ratio)








