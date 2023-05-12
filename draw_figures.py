import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # 2023/03/14? 画loss和gradient的关系图的
# losses = []
# grads = []

# file_name = "ETTh1_96_24_Autoformer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn5_Exp_0"
# with open("./logs_loss_and_grad/"+file_name, "r") as f:
#     for line in f:
#         loss, grad = line.split(',')
#         loss, grad = float(loss), float(grad)
#         losses.append(loss)
#         grads.append(grad)

# print(losses[:10])
# print(grads[:10])

# # losses = losses[:3000]
# # grads = grads[:3000]

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(10, 15), dpi=100)
# plt.scatter(losses, grads, c='red', s=30, label='relationship')
# # plt.xticks(range(2008, 2020, 1))
# # plt.yticks(range(0, 3200, 200))
# plt.xlabel("loss", fontdict={'size': 16})
# plt.ylabel("L2-norm of grad", fontdict={'size': 16})
# plt.title("relationship between loss and L2-norm of grad", fontdict={'size': 20})
# plt.legend(loc='best')
# plt.savefig("./tmp_figures/loss_and_grad_relationship_2.png")
# plt.show()


# # 2023/04/08 画预测值和实际值之间的余弦值角度
# angels = []

# file_name = "ili_36_24_Autoformer_custom_ftM_sl36_ll18_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
# with open("./grad_angels/"+file_name, "r") as f:
#     for line in f:
#         angel = float(line)
#         angels.append(angel)

# print(angels[:10])

# interval = 30
# if interval == 30:
#     ranges = [0, 30, 60, 90, 120, 150, 180]
# elif interval == 15:
#     ranges = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(10, 15), dpi=100)
# n, bins, patches = plt.hist(angels, bins=ranges, edgecolor="r", histtype="bar", alpha=0.5)

# # 2.0 在每个条柱上标注数量
# for i in range(len(n)):
#     plt.text(bins[i]+(bins[1]-bins[0])/2, n[i]*1.01, int(n[i]), ha='center', va='bottom', fontsize=25)
# # 2.1 添加刻度
# min_1 = 0
# max_1 = 180
# division_number = 7 if interval == 30 else 13
# t1 = np.linspace(min_1, max_1, num=division_number)
# plt.xticks(t1)
# # 2.2 添加网格和标题
# plt.grid(linestyle='--')

# plt.xlabel("angel", fontdict={'size': 16})
# plt.ylabel("numbers", fontdict={'size': 16})
# plt.title("the histogram of angels between adapted_gradient and answer_gradient", fontdict={'size': 20})
# # plt.legend(loc='best')
# plt.savefig("./tmp_figures/grad_angels_5_1.png")
# plt.show()




# # 2023/04/14 画经过当前样本做backward之后的模型MSE-Loss的下降值
# diffs = []

# file_name = "answer_lr_0.0001_ETTh1_96_24_Autoformer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_ttn10_Exp_0"
# with open("./loss_diff/"+file_name, "r") as f:
#     for line in f:
#         diff = float(line)
#         diffs.append(diff)

# print(diffs[:10])


# def floatRange(startInt, stopInt, stepInt, precision):
#     f = []
#     for x in range(startInt, stopInt, stepInt):
#         f.append(x/(10**precision))
#     return f

# # ranges = floatRange(-10, 10, 1, 2)
# ranges = floatRange(-50, 50, 5, 2)
# print(ranges)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(10, 15), dpi=100)
# n, bins, patches = plt.hist(diffs, bins=ranges, edgecolor="r", histtype="bar", alpha=0.5)

# # 2.0 在每个条柱上标注数量
# for i in range(len(n)):
#     plt.text(bins[i]+(bins[1]-bins[0])/2, n[i]*1.01, int(n[i]), ha='center', va='bottom', fontsize=25)
# # 2.1 添加刻度
# # min_1 = -0.1
# # max_1 = 0.1
# min_1 = -0.5
# max_1 = 0.5
# division_number = 20+1
# t1 = np.linspace(min_1, max_1, num=division_number)
# plt.xticks(t1)
# # 2.2 添加网格和标题
# plt.grid(linestyle='--')

# plt.xlabel("difference", fontdict={'size': 16})
# plt.ylabel("numbers", fontdict={'size': 16})
# plt.title("the histogram of loss difference before and after\n the backward of the test sample itself,\n positive means that the loss decreases", fontdict={'size': 20})
# # plt.legend(loc='best')
# plt.savefig("./tmp_figures/answer_lr_0.0001_loss_diff_1_2_ETTh1.png")
# plt.show()



# # 2023/04/15 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
# # 这里画的是柱状图，不是直方图

# # # ETTh1
# # mean_error = [0.2825232467669205, 0.3420690088656737, 0.3574817020547605, 0.3705393978093689, 0.3808114281078063, 
# #               0.38586632821474703, 0.3877484064894766, 0.3878066467752339, 0.38774338556708826, 0.38666382334952165, 
# #               0.38308682880917055, 0.37759645628393423, 0.3735901429668319, 0.37039996811169523, 0.36703240360831924, 
# #               0.36498769246547696, 0.3669526425312838, 0.3733200713049876, 0.3817706379181639, 0.3901547758288223, 
# #               0.39469766167581205, 0.3990277771596459, 0.41087240057913954, 0.45379108323071493]
# # ETTm1
# mean_error = [0.25470956661913224, 0.287602646084027, 0.2950821241645256, 0.3040580996432292, 0.313307262633777, 0.32280099611432334, 0.33016007926836827, 0.3370522043719051, 0.3432469475548228, 0.3500065389364098, 0.35779198898003156, 0.36708535603820697, 0.37948391506954027, 0.3864405153877085, 0.39550962074536394, 0.403443534000806, 0.41098701670588716, 0.4187781339499995, 0.4264078834662362, 0.4346426741953037, 0.44423582737959516, 0.45619671798282496, 0.47011285917012857, 0.46704590263277646]

# print(mean_error)

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(10, 15), dpi=100)
# plt.bar(range(len(mean_error)), mean_error)

# # # 2.0 在每个条柱上标注数量
# # for i in range(len(n)):
# #     plt.text(bins[i]+(bins[1]-bins[0])/2, n[i]*1.01, int(n[i]), ha='center', va='bottom', fontsize=25)
# # # 2.1 添加刻度
# # min_1 = 1
# # max_1 = 25
# # division_number = 24+1
# # t1 = np.linspace(min_1, max_1, num=division_number)
# # plt.xticks(t1)
# # # 2.2 添加网格和标题
# # plt.grid(linestyle='--')

# plt.xlabel("index", fontdict={'size': 16})
# plt.ylabel("mean error", fontdict={'size': 16})
# plt.title("Mean error of each index in prediction results,\n that is, we collect each index's error of pred_len and average", fontdict={'size': 20})
# # plt.legend(loc='best')
# plt.savefig("./tmp_figures/error_per_pred_index_2_1_ETTm1.png")
# plt.show()


df = pd.read_csv("./dataset/exchange_rate/exchange_rate.csv", index_col=0)
print(df.head())
print(df.index)