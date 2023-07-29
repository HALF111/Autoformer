import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# KL散度参考：https://zhuanlan.zhihu.com/p/100676922
# 代码实现可参考：https://zhuanlan.zhihu.com/p/143105854
from scipy.stats import entropy, ks_2samp, wasserstein_distance
import scipy

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



pred_len = 96

# train_npy = np.load(f"pl{pred_len}_train.npy")  # (8449, 96, 512)
# val_npy = np.load(f"pl{pred_len}_val.npy")  # (2785, 96, 512)
# test_npy = np.load(f"pl{pred_len}_test.npy")  # (2785, 96, 512)


# train_res, val_res, test_res = [], [], []
# for flag in ["train", "val", "test"]:
#     cur_npy = np.load(f"pl{pred_len}_{flag}.npy")
#     print(cur_npy.shape)

#     # 将timestep和样本个数维度合并
#     # 也即(8449, 96, 512)变成(8449*96, 512)
#     cur_npy = cur_npy.reshape(-1, cur_npy.shape[-1])
#     print(cur_npy.shape)

#     miu = np.sum(cur_npy, axis=0) / cur_npy.shape[0]
#     cov = 0
#     for i in range(cur_npy.shape[0]):
#         diff = (cur_npy[i] - miu).reshape(-1, 1)
#         cov += diff @ diff.T
    
#     print(miu.shape, cov.shape)
    
#     if flag == "train": train_res = [miu, cov]
#     elif flag == "val": val_res = [miu, cov]
#     elif flag == "test": test_res = [miu, cov]

# miu1, miu2, miu3 = train_res[0], val_res[0], test_res[0]
# cov1, cov2, cov3 = train_res[1], val_res[1], test_res[1]

# npy_dir = "./gaussian_params/"
# np.save(npy_dir + "train_miu", miu1)
# np.save(npy_dir + "train_cov", cov1)
# np.save(npy_dir + "val_miu", miu2)
# np.save(npy_dir + "val_cov", cov2)
# np.save(npy_dir + "test_miu", miu3)
# np.save(npy_dir + "test_cov", cov3)


def calc_KL(miu1, cov1, miu2, cov2, n):
    part1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    part2 = np.trace(np.linalg.inv(cov2) @ cov1)
    miu_diff = (miu2 - miu1).reshape(-1, 1)
    part3 = miu_diff.T @ np.linalg.inv(cov2) @ miu_diff

    res = (part1 + part2 + part3 - n) / 2
    # print(part1, part2, miu_diff.shape, part3.shape, res.shape)
    return res


npy_dir = "./gaussian_params/"
miu1, cov1 = np.load(npy_dir+"train_miu.npy"), np.load(npy_dir+"train_cov.npy")
miu2, cov2 = np.load(npy_dir+"val_miu.npy"), np.load(npy_dir+"val_cov.npy")
miu3, cov3 = np.load(npy_dir+"test_miu.npy"), np.load(npy_dir+"test_cov.npy")
miu1, cov1 = miu1.astype("float64"), cov1.astype("float64")
miu2, cov2 = miu2.astype("float64"), cov2.astype("float64")
miu3, cov3 = miu3.astype("float64"), cov3.astype("float64")

print(np.trace(cov1))
print(scipy.linalg.det(cov1))
print(np.linalg.det(cov1))

n = miu1.shape[0]
print("train_val_kldiv", calc_KL(miu1, cov1, miu2, cov2, n))
print("train_test_kldiv", calc_KL(miu1, cov1, miu3, cov3, n))
print("val_test_kldiv", calc_KL(miu2, cov2, miu3, cov3, n))

print("val_train_kldiv", calc_KL(miu2, cov2, miu1, cov1, n))
print("test_train_kldiv", calc_KL(miu3, cov3, miu1, cov1, n))
print("test_val_kldiv", calc_KL(miu3, cov3, miu2, cov2, n))


# print("--------------------------------")
# print("KL_divergence between train and val: ", KL_divergence(train_stat, val_stat))
# print("KL_divergence between train and test: ", KL_divergence(train_stat, test_stat))
# print("KL_divergence between val and test: ", KL_divergence(val_stat, test_stat))
# print("--------------------------------")
# print("KL_divergence2 between train and val: ", KL_divergence2(train_stat, val_stat))
# print("KL_divergence2 between train and test: ", KL_divergence2(train_stat, test_stat))
# print("KL_divergence2 between val and test: ", KL_divergence2(val_stat, test_stat))
# print("--------------------------------")
# print("cross_entropy between train and val: ", cross_entropy(train_stat, val_stat))
# print("cross_entropy between train and test: ", cross_entropy(train_stat, test_stat))
# print("cross_entropy between val and test: ", cross_entropy(val_stat, test_stat))
# print("--------------------------------")
# print("JS_divergence between train and val: ", JS_divergence(train_stat, val_stat))
# print("JS_divergence between train and test: ", JS_divergence(train_stat, test_stat))
# print("JS_divergence between val and test: ", JS_divergence(val_stat, test_stat))
# print("--------------------------------")
# print("KS_test between train and val: ", KS_test(train_stat, val_stat))
# print("KS_test between train and test: ", KS_test(train_stat, test_stat))
# print("KS_test between val and test: ", KS_test(val_stat, test_stat))
# print("--------------------------------")
# print("Wasserstein_distance between train and val: ", Wasserstein_distance(train_stat, val_stat))
# print("Wasserstein_distance between train and test: ", Wasserstein_distance(train_stat, test_stat))
# print("Wasserstein_distance between val and test: ", Wasserstein_distance(val_stat, test_stat))