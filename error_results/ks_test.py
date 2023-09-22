import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import torch
from tqdm import tqdm

# name = "ETTh1"
seq_len = 96
pred_len = 96
device = 'cuda:0'
# seq_len = 24
# pred_len = 24

scale = True
scaler = StandardScaler()

type_map = {'train': 0, 'val': 1, 'test': 2}

def read_data(name, flag):

    # PART I：获取error计算结果
    # PS: how they are calculated?
    # error = pred - true
    # err_mean = np.mean(error)
    # err_var = np.var(error)
    # err_abs_mean = np.mean(np.abs(error))
    # err_abs_var = np.var(np.abs(error))
    # pos_num, neg_num = 0, 0
    # for ei in range(error.shape[0]):
    #     for ej in range(error.shape[1]):
    #         if error[ei][ej] >= 0: pos_num += 1
    #         else: neg_num += 1
    
    
    dir_path = f"./{name}/"
    
    # result = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
    results = []
    with open(dir_path + f"pl{pred_len}_{flag}.txt") as f:
        lines = f.readlines()
        for result in lines:
            result = result.split(",")
            result = [float(item) for item in result]
            results.append(result)
    # print(results[:5])
    
    
    # PART2：获取X和Y原数据
    if "ETT" in name: dataset_dir = f"ETT-small/{name}.csv"
    elif "electricity" in name or "ECL" in name: dataset_dir = f"electricity/electricity.csv"
    elif "Exchange" in name: dataset_dir = "exchange_rate/exchange_rate.csv"
    elif "traffic" in name: dataset_dir = "traffic/traffic.csv"
    elif "weather" in name: dataset_dir = "weather/weather.csv"
    elif "ili" in name or "illness" in name: dataset_dir = "illness/national_illness.csv"
    
    file_name = f"../dataset/{dataset_dir}"
    
    df_raw = pd.read_csv(file_name)
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    
    set_type = type_map[flag]
    
    # 获取train/val/test的划分：
    if "ETTh" in name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif "ETTm" in name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
    
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    
    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values
        
    final_data = data[border1:border2]
    
    data_x, data_y = [], []
    for i in range(len(final_data)-seq_len-pred_len+1):
        cur_x = final_data[i:i+seq_len]
        cur_y = final_data[i+seq_len: i+seq_len+pred_len]
        data_x.append(cur_x)
        data_y.append(cur_y)
    
    # print(data_x[:5])
    # print(data_y[:5])
    
    
    # X数据为data_x, Y数据为data_y, 其他计算结果在results中

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    results = np.asarray(results)
    # print(data_x.shape, data_y.shape, results.shape)
    # plt.clf()
    # plt.hist(results[:, 5], bins=20)
    # plt.savefig(f"temp/{name}-{flag}.png")
    
    return data_x, data_y, results


# for name in ["traffic", "weather", "Exchange", "electricity", "ETTh1", "ETTm2"]:
# pred_len=24
for name in ["Exchange"]:
# for name in ["ETTm2"]:
    # if name != 'exchange':
    #     continue
    dir_path = f"./{name}/"

    data_x_train, data_y_train, results_train = read_data(name, "train")
    data_x_val, data_y_val, results_val = read_data(name, "val")
    data_x_test, data_y_test, results_test = read_data(name, "test")
    print("train/val/test:", data_x_train.shape, data_x_val.shape, data_x_test.shape)

    # metric=5 取的是err_mean这一项
    metric = 5

    error_train = results_train[:, metric]
    # mean = np.mean(error_train)
    # std = np.std(error_train)
    # print(f"mean {mean} std {std}")

    data_x = np.concatenate([data_x_train, data_x_val, data_x_test], axis=0)
    data_y = np.concatenate([data_y_train, data_y_val, data_y_test], axis=0)
    data_x_th = torch.FloatTensor(data_x).reshape((len(data_x), -1)).to(device)
    data_y_th = torch.FloatTensor(data_y).reshape((len(data_y), -1)).to(device)
    results = np.concatenate([results_train, results_val, results_test], axis=0)
    results_th = torch.FloatTensor(results).to(device)
    print(results.shape, results_th.shape)

    offset = len(data_x_train) + len(data_y_val)
    lookback_length = 200 if name == "ili" else 1000
    finetune_count = 10
    
    # ks_value_list = []
    statistic_list = []
    p_value_list = []
    
    
    # 判断各个数据集的周期是多久
    if "ETTh1" in name: period = 24
    elif "ETTh2" in name: period = 24
    elif "ETTm1" in name: period = 96
    elif "ETTm2" in name: period = 96
    elif "electricity" in name: period = 24
    elif "traffic" in name: period = 24
    elif "illness" in name: period = 52.142857
    elif "weather" in name: period = 144
    elif "Exchange" in name: period = 1
    else: period = 1
    
    adapt_cycle = True
    
    # 在这类我们需要先对adaptation样本的x和测试样本的x之间的距离做对比
    import torch.nn.functional as F
            

    # for i in tqdm(range(len(results_test))):
    for i in range(len(results_test)):
            # if i >= 32*5: continue
        # try:
            # 获取当前测试样本
            current_x = data_x_th[i + offset].reshape((1, -1))
            current_y = data_y_th[i + offset].reshape((1, -1))
            
            # 获取lookback window中的样本
            lookback_window_x = data_x_th[offset + i - lookback_length - 1 : offset + i - 1]
            
            distance_pairs = []
            for ii in range(lookback_length):
                # 只对周期性样本计算x之间的距离
                if adapt_cycle:
                    # 为了计算当前的样本和测试样本间时间差是否是周期的倍数
                    # 我们先计算时间差与周期相除的余数
                    if 'ili' in name or 'illness' in name:
                        import math
                        cycle_remainer = math.fmod(lookback_length-1 + pred_len - ii, period)
                    else:
                        cycle_remainer = (lookback_length-1 + pred_len - ii) % period
                    # 定义判定的阈值
                    threshold = period // 12
                    # 如果余数在[-threshold, threshold]之间，那么考虑使用其做fine-tune
                    # 否则的话不将其纳入计算距离的数据范围内
                    if cycle_remainer > threshold or cycle_remainer < -threshold:
                        continue
                
                # 由于lookback_window_x此时为(1000, 96, 862)，已经将每个样本都取出来了，所以此时只需lookback_window_x[ii]即可
                lookback_x = lookback_window_x[ii].reshape(-1)
                
                dist = F.pairwise_distance(current_x, lookback_x, p=2).item()
                distance_pairs.append([ii, dist])

            # 先按距离从小到大排序
            cmp = lambda item: item[1]
            distance_pairs.sort(key=cmp)

            # 筛选出其中最小的selected_data_num个样本出来
            selected_distance_pairs = distance_pairs[:finetune_count]
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            print(f"selected_distance_pairs is: {selected_distance_pairs}")
            
            
            # lookback_window_x = data_x_th[offset + i - lookback_length - 1 : offset + i - 1]
            # distance = torch.norm(lookback_window_x - current_x, dim=1, keepdim=False) / (current_x.shape[1] ** 0.5)
            
            # finetune_inner_index = torch.argsort(distance)[:finetune_count]
            # finetune_idx = finetune_inner_index + offset + i - lookback_length - 1
            # finetune_error = results_th[finetune_idx, metric].cpu().numpy()
            
            # 最后得到fine-tune的样本们的残差值
            finetune_idx = [i + offset - 1 - (lookback_length-1 - idx) for idx in selected_indices]
            finetune_error = results_th[finetune_idx, metric].cpu().numpy()
            print(i)
            print(finetune_error.shape, error_train.shape)
            
            ks_value = stats.kstest(finetune_error, error_train)
            print(ks_value)
            statistic_list.append(ks_value.statistic)
            p_value_list.append(ks_value.pvalue)
        # except:
        #     continue

        # print(ks_value, finetune_error)
    
    print(name, 'diff', np.mean(p_value_list), np.std(p_value_list), np.mean(np.asarray(p_value_list) > 0.05))
        # print(len(results[0]), data_x[0].shape, data_y[0].shape)
        # breakpoint()
        
    file_name = f"./{name}_ks_test.txt"
    with open(file_name, "w") as f:
        for i in range(len(p_value_list)):
            statistic, p_value = statistic_list[i], p_value_list[i]
            f.write(f"{statistic}, {p_value}\n")
    
    data_x_th = data_x_th.cpu()
    data_y_th = data_y_th.cpu()
    results_th = results_th.cpu()
        
        
        
        