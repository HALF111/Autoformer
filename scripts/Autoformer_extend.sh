# python -u run_knn.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model KNN   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 2 --test_train_num 10 --adapted_lr_times 10 --run_train --run_test


# python -u run.py   --is_training 1   --root_path ./dataset/synthetic   --data_path data_rise_0.csv   --model_id rise0_96_96   --model Autoformer   --data rise   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

gpu_num=0
dir_name=Autoformer_extend_result


name=ETTh1
pred_len=96
cur_path=./$dir_name/$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
# Autoformer_extend
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_96   --model Autoformer_extend   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_test > $cur_path/train_and_test.log
for test_train_num in 10
# for test_train_num in 20
do
# for adapted_lr_times in 5 10 20 100
for adapted_lr_times in 0.5 2
do
python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer_extend   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times  --gpu $gpu_num  --run_adapt  > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times.log
done
done
