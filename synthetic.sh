# python -u run_knn.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model KNN   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 2 --test_train_num 10 --adapted_lr_times 10 --run_train --run_test


# python -u run.py   --is_training 1   --root_path ./dataset/synthetic   --data_path data_rise_0.csv   --model_id rise0_96_96   --model Autoformer   --data rise   --features M   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu 1 --test_train_num 10

gpu_num=1
log_dir=./synthetic_results

# # data_period_rise
# log_file=$log_dir'/'PeriodRise.log
# python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_period_rise.csv   --model_id period_rise_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --run_train --run_test > $log_file
# for alpha in 0 1 2
# do
# for lambda_reg in 1000 10000 100000 1000000
# do
# log_file=$log_dir'/'PeriodRise_alpha$alpha'_'lambda$lambda_reg.log
# python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_period_rise.csv   --model_id period_rise_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --test_train_num 20 --lambda_reg $lambda_reg --alpha $alpha --run_calc > $log_file
# done
# done

# period
log_file=$log_dir'/'period.log
# python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_period_0.csv   --model_id period_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --run_train --run_test > $log_file
# for alpha in 0 1 2
for alpha in 1 2
do
for lambda_reg in 1000 10000 100000 1000000
do
log_file=$log_dir'/'period_alpha$alpha'_'lambda$lambda_reg.log
python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_period_0.csv   --model_id period_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --test_train_num 20 --lambda_reg $lambda_reg --alpha $alpha --run_calc > $log_file
done
done

# rise
log_file=$log_dir'/'rise.log
python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_rise_0.csv   --model_id rises_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --run_train --run_test > $log_file
for alpha in 0 1 2
do
for lambda_reg in 1000 10000 100000 1000000
do
log_file=$log_dir'/'rise_alpha$alpha'_'lambda$lambda_reg.log
python -u run.py   --is_training 1   --root_path ./dataset/synthetic/   --data_path data_rise_0.csv   --model_id rises_96_96   --model Autoformer   --data custom   --features S   --seq_len 96   --label_len 48   --pred_len 96   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 1   --c_out 1   --des 'Exp'   --itr 1   --gpu 2  --test_train_num 20 --lambda_reg $lambda_reg --alpha $alpha --run_calc > $log_file
done
done
