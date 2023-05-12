for adapted_lr_times in 1 2 5 10 20 50 100 200 500 1000
do
python -u run.py   --is_training 1   --root_path ./dataset/weather/   --data_path weather.csv   --model_id weather_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 21   --dec_in 21   --c_out 21   --des 'Exp'   --itr 1   --train_epochs 2   --gpu 3  --test_train_num 10 --adapted_lr_times $adapted_lr_times > ./weather_result/weather_result_$adapted_lr_times.txt
done