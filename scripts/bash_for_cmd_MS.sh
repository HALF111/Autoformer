1.1 ETTh1 & pred_len=24
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model Autoformer   --data ETTh1   --features MS   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 1   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

1.2 ETTh1 & pred_len=96
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer   --data ETTh1   --features MS   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 1   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

1.3 ETTh1 & pred_len=168
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_168   --model Autoformer   --data ETTh1   --features MS   --seq_len 96   --label_len 48   --pred_len 168   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 1   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

1.4 ETTh1 & pred_len=720
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_720   --model Autoformer   --data ETTh1   --features MS   --seq_len 96   --label_len 48   --pred_len 720   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 1   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10