# gpu_num=0
# gpu_num=1
gpu_num=2

# dir_name=all_result_batch1
# dir_name=all_result_batch1_average_loss
# dir_name=all_result_batch1_2layer
# dir_name=all_result_batch1_whole_model
# dir_name=all_result_test_train_epochs
# dir_name=all_result_cheated
dir_name=all_result_calc_analytics
# dir_name=all_result_calc_decay_MSE


# # 1.BTC_day数据集
# # 1.1 pred_len=24
# for pred_len in 24
# # for pred_len in 192
# # for pred_len in 192 336 720
# do
# name=BTC_day
# cur_path=./$dir_name/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_day.csv   --model_id BTC_day_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_train --run_test > $cur_path'/'train_and_test_loss.log
# # only run  test first
# # python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_day.csv   --model_id BTC_day_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_test > $cur_path'/'test_loss.log


# for alpha in 0 1 2
# do
# # for test_train_num in 5 10 20 30
# for test_train_num in 10 20
# do
# for lambda_reg in 1000 10000 100000 1000000
# # for lambda_reg in 10000
# # for lambda_reg in 100000 1000000 10000000
# # for lambda_reg in 200000
# do

# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_day.csv   --model_id BTC_day_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --lambda_reg $lambda_reg --alpha $alpha --run_calc > $cur_path'/'alpha$alpha'_'ttn$test_train_num'_'lambda$lambda_reg.log
# done
# done
# done
# done



# # 2.BTC_4_hour数据集
# # 2.1 pred_len=96
# for pred_len in 96
# # for pred_len in 192
# # for pred_len in 192 336 720
# do
# name=BTC_4_hour
# cur_path=./$dir_name/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_4_hour.csv   --model_id BTC_4_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_train --run_test > $cur_path'/'train_and_test_loss.log
# # only run  test first
# # python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_4_hour.csv   --model_id BTC_4_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_test > $cur_path'/'test_loss.log


# for alpha in 0 1 2
# do
# # for test_train_num in 5 10 20 30
# for test_train_num in 10 20
# do
# for lambda_reg in 1000 10000 100000 1000000
# # for lambda_reg in 10000
# # for lambda_reg in 100000 1000000 10000000
# # for lambda_reg in 200000
# do

# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_4_hour.csv   --model_id BTC_4_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --lambda_reg $lambda_reg --alpha $alpha --run_calc > $cur_path'/'alpha$alpha'_'ttn$test_train_num'_'lambda$lambda_reg.log
# done
# done
# done
# done



# # 3.BTC_hour数据集
# # 3.1 pred_len=96
# for pred_len in 96
# # for pred_len in 192
# # for pred_len in 192 336 720
# do
# name=BTC_hour
# cur_path=./$dir_name/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_hour.csv   --model_id BTC_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_train --run_test > $cur_path'/'train_and_test_loss.log
# # only run  test first
# # python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_hour.csv   --model_id BTC_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_test > $cur_path'/'test_loss.log


# for alpha in 0 1 2
# do
# # for test_train_num in 5 10 20 30
# for test_train_num in 10 20
# do
# for lambda_reg in 1000 10000 100000 1000000
# # for lambda_reg in 10000
# # for lambda_reg in 100000 1000000 10000000
# # for lambda_reg in 200000
# do

# python -u run.py   --is_training 1   --root_path ./dataset/KNF/Cryptos/   --data_path BTC_hour.csv   --model_id BTC_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 8   --dec_in 8   --c_out 8   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --lambda_reg $lambda_reg --alpha $alpha --run_calc > $cur_path'/'alpha$alpha'_'ttn$test_train_num'_'lambda$lambda_reg.log
# done
# done
# done
# done



# 5.BTC_Price_hour数据集
# 5.1 pred_len=96
for pred_len in 96
# for pred_len in 192 336 720
do
name=BTC_Price_hour
cur_path=./$dir_name/$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
# run train and test first
python -u run.py   --is_training 1   --root_path ./dataset/My_BTC_data/   --data_path BTC_Price_hour.csv   --model_id BTC_Price_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 4   --dec_in 4   --c_out 4   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_train --run_test > $cur_path'/'train_and_test_loss.log
# only run  test first
# python -u run.py   --is_training 1   --root_path ./dataset/My_BTC_data/   --data_path BTC_Price_hour.csv   --model_id BTC_Price_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 4   --dec_in 4   --c_out 4   --des 'Exp'   --itr 1   --gpu $gpu_num  --run_test > $cur_path'/'test_loss.log


for alpha in 0 1 2
do
# for test_train_num in 5 10 20 30
for test_train_num in 10 20
do
for lambda_reg in 1000 10000 100000 1000000
# for lambda_reg in 10000
# for lambda_reg in 100000 1000000 10000000
# for lambda_reg in 200000
do

python -u run.py   --is_training 1   --root_path ./dataset/My_BTC_data/   --data_path BTC_Price_hour.csv   --model_id BTC_Price_hour_96_$pred_len   --model Autoformer   --data custom   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 4   --dec_in 4   --c_out 4   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --lambda_reg $lambda_reg --alpha $alpha --run_calc > $cur_path'/'alpha$alpha'_'ttn$test_train_num'_'lambda$lambda_reg.log
done
done
done
done
