import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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



pred_len = 96

# train_npy = np.load(f"pl{pred_len}_train.npy")
# val_npy = np.load(f"pl{pred_len}_val.npy")
# test_npy = np.load(f"pl{pred_len}_test.npy")
train_npy = np.load(f"pl{pred_len}_train.npy")  # (8449, 96, 512)
val_npy = np.load(f"pl{pred_len}_val.npy")  # (2785, 96, 512)
test_npy = np.load(f"pl{pred_len}_test.npy")  # (2785, 96, 512)
print(train_npy.shape, val_npy.shape, test_npy.shape)


# 1.将timestep和样本个数维度合并
# 也即(8449, 96, 512)变成(8449*96, 512)
train_npy = train_npy.reshape(-1, train_npy.shape[-1])
val_npy = val_npy.reshape(-1, val_npy.shape[-1])
test_npy = test_npy.reshape(-1, test_npy.shape[-1])
print(train_npy.shape, val_npy.shape, test_npy.shape)

train_val_kldivs, train_test_kldivs, val_test_kldivs = [], [], []
# val_train_kldivs, test_train_kldivs, test_val_kldivs = [], [], []
for channel in range(train_npy.shape[-1]):
# for channel in range(10):
    # 取出对应channel的维度
    train_res, val_res, test_res = train_npy[:, channel], val_npy[:, channel], test_npy[:, channel]

    max_value = max(max(train_res), max(val_res), max(test_res))
    min_value = min(min(train_res), min(val_res), min(test_res))
    print("max:", max(train_res), max(val_res), max(test_res), max_value)
    print("min:", min(train_res), min(val_res), min(test_res), min_value)


    step_num = 51
    interval = (max_value - min_value) / step_num
    x = np.arange(min_value, max_value+1.5*interval, interval)
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

    train_stat = train_stat / len(train_res)
    val_stat = val_stat / len(val_res)
    test_stat = test_stat / len(test_res)
    EPS = 1e-9
    train_stat += EPS
    val_stat += EPS
    test_stat += EPS
    print("train_stat: " + str(train_stat))
    print("val_stat: " + str(val_stat))
    print("test_stat: " + str(test_stat))

    train_val = KL_divergence(train_stat, val_stat)
    train_test = KL_divergence(train_stat, test_stat)
    val_test = KL_divergence(val_stat, test_stat)
    print(train_val, train_test, val_test)
    train_val_kldivs.append(train_val)
    train_test_kldivs.append(train_test)
    val_test_kldivs.append(val_test)

    # val_train = KL_divergence(val_stat, train_stat)
    # test_train = KL_divergence(test_stat, train_stat)
    # test_val = KL_divergence(test_stat, val_stat)
    # print(val_train, test_train, test_val)
    # val_train_kldivs.append(val_train)
    # test_train_kldivs.append(test_train)
    # test_val_kldivs.append(test_val)


x = [i for i in range(train_npy.shape[-1])]
# x = [i for i in range(10)]
plt.figure()
plt.title("KL divergence of ETTh1 for each channel")
plt.xlabel("channel")
plt.ylabel("KL divergence")
plt.plot(x, train_val_kldivs, label="train_val")
plt.plot(x, train_test_kldivs, label="train_test")
plt.plot(x, val_test_kldivs, label="val_test")
# plt.plot(x, val_train_kldivs, label="val_train")
# plt.plot(x, test_train_kldivs, label="test_train")
# plt.plot(x, test_val_kldivs, label="test_val")
plt.legend(loc="lower right")
plt.savefig("./tmp.pdf")


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