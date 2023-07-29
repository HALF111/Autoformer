import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
# TODO:这里应当化成(8449, 96*512)更好，还是原来的(8449*96, 512)更好？


# tsne = TSNE(n_components=2)
# train_tsne = tsne.fit_transform(train_res)
# print(tsne.embedding_)


res_dir = "./k_means_results/"
SSE = []
# n_clusters = 24
# for n_clusters in [4, 6, 12, 24]: 
for n_clusters in [48, 96]: 
    k_means = KMeans(n_clusters=n_clusters, random_state=0)
    # k_means = KMeans(n_clusters=n_clusters, random_state=0)

    train_clusters = k_means.fit(train_res)
    print(k_means.labels_)
    print(k_means.cluster_centers_)
    print(k_means.inertia_)
    SSE.append([n_clusters, k_means.inertia_])
    
    np.save(res_dir + f"k={n_clusters}_labels.npy", np.array(k_means.labels_))
    np.save(res_dir + f"k={n_clusters}_cluster_centers.npy", np.array(k_means.cluster_centers_))

print(SSE)

