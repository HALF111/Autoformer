for adapted_lr_times in 1 2 5 10 20 50 100 200 500 1000
do
python -u run.py   --is_training 1   --root_path ./dataset/traffic/   --data_path traffic.csv   --model_id traffic_96_96   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 862   --dec_in 862   --c_out 862   --des 'Exp'   --itr 1   --train_epochs 3   --gpu 3  --test_train_num 10 --adapted_lr_times $adapted_lr_times > ./traffic_result/traffic_result_$adapted_lr_times.txt
done