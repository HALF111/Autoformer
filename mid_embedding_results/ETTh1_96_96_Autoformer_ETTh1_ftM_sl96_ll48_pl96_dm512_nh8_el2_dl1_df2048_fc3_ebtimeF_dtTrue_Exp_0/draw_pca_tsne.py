import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

pred_len = 96


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



# 降维
n_clusters = 4
res_dir = "./k_means_results/"
label_file = res_dir + f"k={n_clusters}_labels.npy"
Y = np.load(label_file)
print(Y.shape)

X = train_res
print(X.shape)

# 核心方法：PCA或TSNE！！
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
X_pca = X_tsne

print(X_pca.shape)

X_pca = np.vstack((X_pca.T, Y)).T
print(X_pca.shape)
print(X_pca[:30])


df_pca = pd.DataFrame(X_pca, columns=["dim_1", "dim_2", "label"])
print(df_pca.head())

plt.figure()
sns.scatterplot(data=df_pca, hue='label', x='dim_1', y='dim_2')
# for i in range(n_clusters):
#     tmp_df = df_pca.loc[df_pca['label'] == i, ["dim_1", "dim_2"]]
#     plt.scatter(tmp_df["dim_1"], tmp_df["dim_2"])
# plt.legend()
plt.savefig(f"k={n_clusters}_tsne.pdf")




