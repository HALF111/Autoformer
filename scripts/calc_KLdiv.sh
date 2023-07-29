
dir_name=KL_div_results

# 1.1 ETTh1 & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'ETTh1.log


# 2.1 ETTh2 & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh2.csv   --model_id ETTh2_96_96   --model Autoformer   --data ETTh2   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'ETTh2.log


# 3.1 ETTm1 & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm1.csv   --model_id ETTm1_96_96   --model Autoformer   --data ETTm1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'ETTm1.log



# 4.1 ETTm2 & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTm2.csv   --model_id ETTm2_96_96   --model Autoformer   --data ETTm2   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'ETTm2.log


# 5.1 Electricity & pred_len=96
python -u run.py  --is_training 1   --root_path ./dataset/electricity/   --data_path electricity.csv   --model_id ECL_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3  --enc_in 321   --dec_in 321   --c_out 321   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'ECL.log


# 6.1 Exhange & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/exchange_rate/   --data_path exchange_rate.csv   --model_id Exchange_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'Exchange.log


# 7.1 Traffic & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/traffic/   --data_path traffic.csv   --model_id traffic_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --des 'Exp'   --itr 1   --train_epochs 3   --gpu 1  --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'Traffic.log


# 8.1 Weather & pred_len=96
# python -u run.py   --is_training 1   --root_path ./dataset/weather/   --data_path weather.csv   --model_id weather_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'   --itr 1   --train_epochs 2   --gpu 1  --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'Weather.log


# 9.1 Illness & pred_len=24
# python -u run.py   --is_training 1   --root_path ./dataset/illness/   --data_path national_illness.csv   --model_id ili_36_24   --model Autoformer   --data custom   --features M   --seq_len 36   --label_len 18   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1  --test_train_num 10  --run_calc_kldiv --batch_size 1 > $dir_name'/'Illness.log

