import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# name = "ETTh1"
seq_len = 96
pred_len = 96

scale = False
scaler = StandardScaler()

type_map = {'train': 0, 'val': 1, 'test': 2}


# for name in ["ETTh1", "electricity"]:
for name in ["ETTh1", "electricity"]:
    dir_path = f"./{name}/"

    for flag in ["train", "val", "test"]:
        
        # PART I：获取error计算结果
        # error = pred - true
        # result = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
        results = []
        with open(dir_path + f"pl{pred_len}_{flag}.txt") as f:
            lines = f.readlines()
            for result in lines:
                result = result.split(",")
                result = [float(item) for item in result]
                results.append(result)
        print(results[:5])
        
        
        # # PART2：获取X和Y原数据
        # df_raw = pd.read_csv(f"../dataset/ETT-small/{name}.csv")
        # cols_data = df_raw.columns[1:]
        # df_data = df_raw[cols_data]
        
        # set_type = type_map[flag]
        
        # # 获取train/val/test的划分：
        # if "ETTh" in name:
        #     border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        #     border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # elif "ETTm" in name:
        #     border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        #     border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        # else:
        #     num_train = int(len(df_raw) * 0.7)
        #     num_test = int(len(df_raw) * 0.2)
        #     num_vali = len(df_raw) - num_train - num_test
        #     border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        #     border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        # border1 = border1s[set_type]
        # border2 = border2s[set_type]
        
        # if scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     scaler.fit(train_data.values)
        #     data = scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
            
        # final_data = data[border1:border2]
        
        # data_x, data_y = [], []
        # for i in range(len(final_data)-seq_len-pred_len+1):
        #     cur_x = final_data[i:i+seq_len]
        #     cur_y = final_data[i+seq_len: i+seq_len+pred_len]
        #     data_x.append(cur_x)
        #     data_y.append(cur_y)
        
        # # print(data_x[:5])
        # # print(data_y[:5])
        
        # print(len(results), len(data_x), len(data_y))
        # print(len(results[0]), data_x[0].shape, data_y[0].shape)
        
        
        
        