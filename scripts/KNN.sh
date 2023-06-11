# gpu_num=0
# gpu_num=1
gpu_num=2

# dir_name=all_result_batch1
# dir_name=all_result_batch1_average_loss
# dir_name=all_result_batch1_2layer
# dir_name=all_result_batch1_whole_model
dir_name=all_result_knn


# 1.ETTh1数据集
# 1.1 pred_len=24
# for pred_len in 24 96 192 336 720
for pred_len in 96
# for pred_len in 336 720
do
name=ETTh1
cur_path=./$dir_name/$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi

for feature_dim in 50 100 200
do
for k_value in 2 5 10 20
do

python -u run_knn.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model KNN   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test  --feature_dim $feature_dim  --k_value $k_value  >  $cur_path'/'feat_dim$feature_dim'_'k_value$k_value.log

done
done
done