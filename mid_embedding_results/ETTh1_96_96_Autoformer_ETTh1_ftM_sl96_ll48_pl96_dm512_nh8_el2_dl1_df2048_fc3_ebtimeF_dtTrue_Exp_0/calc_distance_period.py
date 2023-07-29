import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

pred_len = 96


# train_npy = np.load(f"pl{pred_len}_train.npy")  # (8449, 96, 512)
# val_npy = np.load(f"pl{pred_len}_val.npy")  # (2785, 96, 512)
# test_npy = np.load(f"pl{pred_len}_test.npy")  # (2785, 96, 512)
# print(train_npy.shape, val_npy.shape, test_npy.shape)
# # (8449, 96, 512), (2785, 96, 512), (2785, 96, 512)


# train_res, val_res, test_res = [], [], []
# for flag in ["train", "val", "test"]:
#     cur_npy = np.load(f"pl{pred_len}_{flag}.npy")
#     print(cur_npy.shape)

#     # 将timestep和样本个数维度合并
#     # 也即(8449, 96, 512)变成(8449*96, 512)
#     cur_npy = cur_npy.reshape(-1, cur_npy.shape[-1])
#     print(cur_npy.shape)
    
#     if flag == "train": train_res = cur_npy
#     elif flag == "val": val_res = cur_npy
#     elif flag == "test": test_res = cur_npy


# print(train_res.shape, val_res.shape, test_res.shape)
# # (8449*96, 512), (2785*96, 512), (2785*96, 512)


train_res, val_res, test_res = [], [], []
for flag in ["train", "val", "test"]:
    cur_npy = np.load(f"pl{pred_len}_{flag}.npy")
    print(cur_npy.shape)

    # 将timestep和样本个数维度合并
    # 也即(8449, 96, 512)变成(8449, 96*512)
    # TODO:这里化成(8449*96, 512)还是(8449, 96*512)更好？
    # cur_npy = cur_npy.reshape(-1, cur_npy.shape[-1])
    cur_npy = cur_npy.reshape(cur_npy.shape[0], -1)
    print(cur_npy.shape)
    
    if flag == "train": train_res = cur_npy
    elif flag == "val": val_res = cur_npy
    elif flag == "test": test_res = cur_npy


print(train_res.shape, val_res.shape, test_res.shape)
# (8449, 96*512), (2785, 96*512), (2785, 96*512)
# TODO:这里应当化成(8449, 96*512)更好，还是(8449*96, 512)更好？


start = 1000
for i in range(start+1, start+101):
    print(i-start, np.linalg.norm(train_res[i]-train_res[start]))

start = 1000
for i in range(start+1, start+101):
    print(i-start, np.linalg.norm(val_res[i]-val_res[start]))

start = 1000
for i in range(start+1, start+101):
    print(i-start, np.linalg.norm(test_res[i]-test_res[start]))