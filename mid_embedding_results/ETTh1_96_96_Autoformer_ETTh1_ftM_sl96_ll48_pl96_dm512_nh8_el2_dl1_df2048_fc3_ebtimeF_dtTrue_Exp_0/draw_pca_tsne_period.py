import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

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



# res_dir = "./k_means_results/"
# SSE = []
# # n_clusters = 24
# # for n_clusters in [4, 6, 12, 24]: 
# # for n_clusters in [48, 96]: 
# for n_clusters in [4]: 
#     k_means = KMeans(n_clusters=n_clusters, random_state=0)
#     # k_means = KMeans(n_clusters=n_clusters, random_state=0)

#     train_clusters = k_means.fit(train_res)
#     print(k_means.labels_)
#     print(k_means.cluster_centers_)
#     print(k_means.inertia_)
#     SSE.append([n_clusters, k_means.inertia_])
    
#     # np.save(res_dir + f"k={n_clusters}_labels.npy", np.array(k_means.labels_))
#     # np.save(res_dir + f"k={n_clusters}_cluster_centers.npy", np.array(k_means.cluster_centers_))

# print(SSE)


# PCA或T-SNE降维
n_clusters = 4
# res_dir = "./k_means_results/"
# label_file = res_dir + f"k={n_clusters}_labels.npy"
# Y = np.load(label_file)
# print(Y.shape)

X = train_res
print(X.shape)

# 核心方法：PCA或TSNE！！
# * 下面的代码二选一注释掉
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X)
# X_pca = X_tsne

print(X_pca.shape)


# n_clusters = 24
n_clusters = 6
period = 24
data_len = X_pca.shape[0]
Y = np.zeros((data_len), dtype=int)
for i in range(0, data_len, period//n_clusters):
    Y[i] = i % period
    for k in range(1, period//n_clusters):
        if i+k >= data_len: break
        Y[i+k] = Y[i]
    # i += period//n_clusters - 1
print(Y.shape)
print(Y[:30])
print(Y[-30:])

X_pca = np.vstack((X_pca.T, Y)).T
print(X_pca.shape)
print(X_pca[:30])


df_pca = pd.DataFrame(X_pca, columns=["dim_1", "dim_2", "label"])
print(df_pca.head())

plt.figure()
# sns.scatterplot(data=df_pca, hue='label', x='dim_1', y='dim_2')
# for i in range(n_clusters):
#     tmp_df = df_pca.loc[df_pca['label'] == i, ["dim_1", "dim_2"]]
#     plt.scatter(tmp_df["dim_1"], tmp_df["dim_2"])
for i in range(0, period, period//n_clusters):
    print(i)
    tmp_df = df_pca.loc[df_pca['label'] == i, ["dim_1", "dim_2"]]
    plt.scatter(tmp_df["dim_1"], tmp_df["dim_2"], s=15)
plt.legend()
plt.savefig(f"period{period}_{n_clusters}clusters_pca.pdf")