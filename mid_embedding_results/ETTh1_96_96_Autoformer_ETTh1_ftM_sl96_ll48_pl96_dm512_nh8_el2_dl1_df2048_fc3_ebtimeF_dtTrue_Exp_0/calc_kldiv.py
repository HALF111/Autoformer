import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

pred_len = 96

# train_npy = np.load(f"pl{pred_len}_train.npy")
# val_npy = np.load(f"pl{pred_len}_val.npy")
# test_npy = np.load(f"pl{pred_len}_test.npy")

# print(train_npy.shape, val_npy.shape, test_npy.shape)
# # (8449, 96, 512), (2785, 96, 512), (2785, 96, 512)

train_res, val_res, test_res = [], [], []
for flag in ["train", "val", "test"]:
    cur_npy = np.load(f"pl{pred_len}_{flag}.npy")
    print(cur_npy.shape)

    # total_length = cur_npy.shape[0] + cur_npy.shape[1] - 1
    result_list = []
    for i in range(cur_npy.shape[0] + cur_npy.shape[1] - 1):
        # 先获取当前需要累加的个数
        if i < cur_npy.shape[1]:
            total_num = i + 1
        elif i >= cur_npy.shape[1] and i < cur_npy.shape[0]:
            total_num = cur_npy.shape[1]
        else:
            total_num = cur_npy.shape[0] + cur_npy.shape[1] - 1 - i
        
        result = 0
        if i < cur_npy.shape[0]:
            for j in range(0, total_num):
                result += cur_npy[i-j][j]
        else:
            for j in range(i+1 - cur_npy.shape[0], cur_npy.shape[1]):
                result += cur_npy[i-j][j]
        
        result /= total_num
        # print(results.shape)
        result_list.append(result)
    
    # print(len(result_list))
    result_list = np.array(result_list)
    print(result_list.shape)
    if flag == "train": train_res = result_list
    elif flag == "val": val_res = result_list
    elif flag == "test": test_res = result_list


# 1.最简单做法-求和
train_res, val_res, test_res = train_res.sum(axis=1), val_res.sum(axis=1), test_res.sum(axis=1)
print(train_res.shape, val_res.shape, test_res.shape)

max_value = max(max(train_res), max(val_res), max(test_res))
min_value = min(min(train_res), min(val_res), min(test_res))
print(max(train_res), max(val_res), max(test_res), max_value)
print(min(train_res), min(val_res), min(test_res), min_value)


step_num = 21
interval = (max_value - min_value) / step_num
x = np.arange(min_value, max_value+interval, interval)
print(x)

# 统计各个区间内的数据值信息
train_stat = np.histogram(a=train_res, bins=x)[0]
val_stat = np.histogram(a=val_res, bins=x)[0]
test_stat = np.histogram(a=test_res, bins=x)[0]
print("train_statistics_num: " + str(train_stat))
print("val_statistics_num: " + str(val_stat))
print("test_statistics_num: " + str(test_stat))

print(sum(train_stat), len(train_res))
assert sum(train_stat) == len(train_res)

train_stat = train_stat/len(train_res)
val_stat = val_stat / len(val_res)
test_stat = test_stat / len(test_res)
EPS = 1e-6
train_stat += EPS
val_stat += EPS
test_stat += EPS
print("train_stat: " + str(train_stat))
print("val_stat: " + str(val_stat))
print("test_stat: " + str(test_stat))


# KL散度参考：https://zhuanlan.zhihu.com/p/100676922
# 代码实现可参考：https://zhuanlan.zhihu.com/p/143105854
from scipy.stats import entropy, ks_2samp, wasserstein_distance

def KL_divergence(p, q):
    return entropy(p, q)

def KL_divergence2(p, q):
    KL = 0.0
    px, qx = p / np.sum(p), q / np.sum(q)
    for i in range(len(px)):
        KL += px[i] * np.log(px[i] / qx[i])
    return KL

# 交叉熵
def cross_entropy(p, q):
    p = np.float_(p)
    q = np.float_(q)
    return -np.sum([p[i] * np.log2(q[i]) for i in range(len(p))])

# JS散度
def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * entropy(p, M) + 0.5 * entropy(q, M)

# KS检测方法
def KS_test(p, q):
    res = ks_2samp(p, q).pvalue
    return res

# Wasserstein距离
def Wasserstein_distance(p, q):
    dist = wasserstein_distance(p, q)
    return dist

print("--------------------------------")
print("KL_divergence between train and val: ", KL_divergence(train_stat, val_stat))
print("KL_divergence between train and test: ", KL_divergence(train_stat, test_stat))
print("KL_divergence between val and test: ", KL_divergence(val_stat, test_stat))
print("--------------------------------")
print("KL_divergence2 between train and val: ", KL_divergence2(train_stat, val_stat))
print("KL_divergence2 between train and test: ", KL_divergence2(train_stat, test_stat))
print("KL_divergence2 between val and test: ", KL_divergence2(val_stat, test_stat))
print("--------------------------------")
print("cross_entropy between train and val: ", cross_entropy(train_stat, val_stat))
print("cross_entropy between train and test: ", cross_entropy(train_stat, test_stat))
print("cross_entropy between val and test: ", cross_entropy(val_stat, test_stat))
print("--------------------------------")
print("JS_divergence between train and val: ", JS_divergence(train_stat, val_stat))
print("JS_divergence between train and test: ", JS_divergence(train_stat, test_stat))
print("JS_divergence between val and test: ", JS_divergence(val_stat, test_stat))
print("--------------------------------")
print("KS_test between train and val: ", KS_test(train_stat, val_stat))
print("KS_test between train and test: ", KS_test(train_stat, test_stat))
print("KS_test between val and test: ", KS_test(val_stat, test_stat))
print("--------------------------------")
print("Wasserstein_distance between train and val: ", Wasserstein_distance(train_stat, val_stat))
print("Wasserstein_distance between train and test: ", Wasserstein_distance(train_stat, test_stat))
print("Wasserstein_distance between val and test: ", Wasserstein_distance(val_stat, test_stat))