1.1 ETTh1 & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

1.2 ETTh1 & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 200 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5  --adapt_cycle

1.3 ETTh1 & pred_len=168
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_168   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 168   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

1.4 ETTh1 & pred_len=720
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_720   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 720   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10



2.1 ETTh2 & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --model_id ETTh2_96_24   --model Autoformer   --data ETTh2   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

2.2 ETTh2 & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --model_id ETTh2_96_96   --model Autoformer   --data ETTh2   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --model_id ETTh2_96_96   --model Autoformer   --data ETTh2   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle



3.1 ETTm1 & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --model_id ETTm1_96_24   --model Autoformer   --data ETTm1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --model_id ETTm1_96_24   --model Autoformer   --data ETTm1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 500  --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5



4.1 ETTm2 & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --model_id ETTm2_96_24   --model Autoformer   --data ETTm2   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --model_id ETTm2_96_24   --model Autoformer   --data ETTm2   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --model_id ETTm2_96_24   --model Autoformer   --data ETTm2   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle



5.1 Electricity & pred_len=96
python -u run.py  --is_training 1   --root_path ./dataset/electricity/   --data_path electricity.csv   --model_id ECL_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3  --enc_in 321   --dec_in 321   --c_out 321   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10
python -u run.py  --is_training 1   --root_path ./dataset/electricity/   --data_path electricity.csv   --model_id ECL_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3  --enc_in 321   --dec_in 321   --c_out 321   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 500 --adapt_cycle



6.1 Exchange & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --model_id Exchange_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --model_id Exchange_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle



7.1 Traffic & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/traffic/   --data_path traffic.csv   --model_id traffic_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --des 'Exp'   --itr 1   --train_epochs 3   --gpu 1  --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/traffic/   --data_path traffic.csv   --model_id traffic_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --des 'Exp'   --itr 1   --train_epochs 3   --gpu 1  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 10 --adapt_cycle



8.1 Weather & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/weather/   --data_path weather.csv   --model_id weather_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'   --itr 1   --train_epochs 2   --gpu 1  --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/weather/   --data_path weather.csv   --model_id weather_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'   --itr 1   --train_epochs 2   --gpu 1  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 10 --adapt_cycle


9.1 Illness & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/illness/   --data_path national_illness.csv   --model_id ili_36_24   --model Autoformer   --data custom   --features M   --seq_len 36   --label_len 18   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10
python -u run.py   --is_training 1   --root_path ./dataset/illness/   --data_path national_illness.csv   --model_id ili_36_24   --model Autoformer   --data custom   --features M   --seq_len 36   --label_len 18   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 200 --run_select_with_distance --selected_data_num 30 --adapted_lr_times 10



10.1 BTC_price_day & pred_len=24
# BTC_day
python -u run.py   --is_training 1   --root_path ./dataset/   --data_path BTC_day.csv   --model_id BTC_day_96_24   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48  --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10
# BTC_4_hour
python -u run.py   --is_training 1   --root_path ./dataset/   --data_path BTC_4_hour.csv   --model_id BTC_4_hour_96_24   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48  --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10
# BTC_hour
python -u run.py   --is_training 1   --root_path ./dataset/   --data_path BTC_hour.csv   --model_id BTC_hour_96_24   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48  --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10