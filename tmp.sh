# python -u run_knn.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model KNN   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 2 --test_train_num 10 --adapted_lr_times 10 --run_train --run_test


# python -u run.py   --is_training 1   --root_path ./dataset/synthetic   --data_path data_rise_0.csv   --model_id rise0_96_96   --model Autoformer   --data rise   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

gpu_num=1
# # 01.draw_error on ETTh1
# # 注意batch_size设置为1，以及--run_test --run_adapt
# python -u draw_error.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num 10 --adapted_lr_times 5 --batch_size 1  --run_adapt

# # 02.ETTm2
# python -u draw_error.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --model_id ETTm2_96_96   --model Autoformer   --data ETTm2   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num 10 --adapted_lr_times 5 --batch_size 1  --run_adapt

# 03.Electricity
python -u draw_error.py  --is_training 1   --root_path ./dataset/electricity/   --data_path electricity.csv   --model_id ECL_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3  --enc_in 321   --dec_in 321   --c_out 321   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num 10 --adapted_lr_times 5000 --batch_size 1 --run_adapt --figure_name ECL

# 04.Exchange
python -u draw_error.py   --is_training 1   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --model_id Exchange_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --test_train_num 10 --adapted_lr_times 200 --batch_size 1  --run_adapt --figure_name Exchange

# 05.Traffic
python -u draw_error.py   --is_training 1   --root_path ./dataset/traffic/   --data_path traffic.csv   --model_id traffic_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --des 'Exp'   --itr 1   --train_epochs 3   --gpu $gpu_num  --test_train_num 10 --adapted_lr_times 10000 --batch_size 1 --run_adapt --figure_name Traffic

# 06.Weather
python -u draw_error.py   --is_training 1   --root_path ./dataset/weather/   --data_path weather.csv   --model_id weather_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'   --itr 1   --train_epochs 2   --gpu $gpu_num  --test_train_num 10 --adapted_lr_times 10 --batch_size 1  --run_adapt --figure_name Weather

# 07.Illness
python -u draw_error.py   --is_training 1   --root_path ./dataset/illness/   --data_path national_illness.csv   --model_id ili_36_24   --model Autoformer   --data custom   --features M   --seq_len 36   --label_len 18   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10 --adapted_lr_times 100 --batch_size 1  --run_adapt --figure_name ILI