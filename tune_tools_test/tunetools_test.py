# train.py
from tunetools import decorator, Parameter

# This is the main function, which will be executed during training. 
# Tunetools will recognize the parameters automatically, and construct 
# grid search with the given domains. 
# "num_sample=2" means each experiment will run for 2 times.
# @decorator.main(num_sample=2)
@decorator.main(num_sample=1)
def main(
        # Register the hyper-parameters manifest here. 
        alpha: Parameter(default=0, domain=[0, 1, 2]),
        test_train_num: Parameter(default=10, domain=[20, 30]),
        lambda_reg: Parameter(default=10000, domain=[1000, 10000, 100000, 1000000]),

        # model: Parameter(default="Autoformer", domain=["Autoformer", "FEDformer"]),
        model: Parameter(default="Autoformer", domain=["Autoformer"]),

        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ECL", "Traffic", "Exchange", "ETTh2", "ETTm1", "ETTm2", "Weather", "Illness"]),
        dataset: Parameter(default="ETTh1", domain=["ETTh1", "ECL", "Traffic", "Exchange", "ETTm2", "Weather"]),

        # pred_len: Parameter(default=96, domain=[96, 196, 336, 720])
        pred_len: Parameter(default=96, domain=[96]),
        gpu: Parameter(default=0, domain=[0]),

):
    # Do training here, use all the parameters...
    import os
    import random

    mapping = {
        "ETTh1": {"root_path": "./dataset/ETT-small", "data_path": "ETTh1", "data": "ETTh1", "model_id_name": "ETTh1", "variant_num": 7},
        "ETTh2": {"root_path": "./dataset/ETT-small", "data_path": "ETTh2", "data": "ETTh2", "model_id_name": "ETTh2", "variant_num": 7},
        "ETTm1": {"root_path": "./dataset/ETT-small", "data_path": "ETTm1", "data": "ETTm1", "model_id_name": "ETTm1", "variant_num": 7},
        "ETTm2": {"root_path": "./dataset/ETT-small", "data_path": "ETTm2", "data": "ETTm2", "model_id_name": "ETTm2", "variant_num": 7},

        "ECL": {"root_path": "./dataset/electricity", "data_path": "electricity", "data": "custom", "model_id_name": "ECL", "variant_num": 321},
        "Exchange": {"root_path": "./dataset/exchange_rate", "data_path": "exchange_rate", "data": "custom", "model_id_name": "Exchange", "variant_num": 8},
        "Weather": {"root_path": "./dataset/weather", "data_path": "weather", "data": "custom", "model_id_name": "weather", "variant_num": 21},
        "Traffic": {"root_path": "./dataset/traffic", "data_path": "traffic", "data": "custom", "model_id_name": "traffic", "variant_num": 862},
        # "Illness": {"root_path": "./dataset/illness", "data_path": "national_illness", "data": "custom", "model_id_name": "ili", "variant_num": 7},
    }
    
    root_path = mapping[dataset]["root_path"]
    data_path = mapping[dataset]["data_path"]
    data = mapping[dataset]["data"]
    model_id_name = mapping[dataset]["model_id_name"]
    variant_num = mapping[dataset]["variant_num"]
    seq_len, label_len = 96, 48

    log_path = f"./all_result_calc_analytics/{dataset}_pl{pred_len}"
    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    
    os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
              --data_path {data_path}.csv   --model_id {model_id_name}_{seq_len}_{pred_len}   \
              --model {model}   --data {data}   --features M   \
              --seq_len {seq_len}   --label_len {label_len}   --pred_len {pred_len}   \
              --e_layers 2   --d_layers 1   --factor 3   \
              --enc_in {variant_num}   --dec_in {variant_num}   \
              --c_out {variant_num}   --des 'Exp'   --itr 1   --train_epochs 1   \
              --gpu {gpu}  --test_train_num {test_train_num}  --lambda_reg {lambda_reg} --alpha {alpha} \
              --run_calc > {log_path}/alpha{alpha}_ttn{test_train_num}_lambda{lambda_reg}.log")

    # 读取mse和mae结果
    result_dir = "./mse_and_mae_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # dataset_name = data_path.replace(".csv", "")
    dataset_name = data_path
    file_name = f"{dataset_name}_pl{pred_len}_alpha{alpha}_ttn{test_train_num}_lambda{lambda_reg}.txt"
    file_path = os.path.join(result_dir, file_name)
    with open(file_path) as f:
        line = f.readline()
        line = line.split(",")
        mse, mae = float(line[0]), float(line[1])

    return {
        "mse": mse,
        "mae": mae,
    }

# @decorator.filtering
# def filter_func(alpha, test_train_num, lambda_reg, model, dataset, gpu):
#     # Filter some parameter combinations you don't want to use.
#     return dataset != 'd3'