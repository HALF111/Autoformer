if [ ! -d "./logs_test" ]; then
    mkdir ./logs_test
fi

model_name=Autoformer

### M

test_train_num=10

for preLen in 96
do

# # ETTh1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_$preLen \
#   --model Autoformer \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --test_train_num $test_train_num \
#   --gpu 2 \
#   > logs_test/$model_name'_'M_ETTh1_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# # ETTh2
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_$preLen \
#   --model Autoformer \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --test_train_num $test_train_num \
#   --gpu 2 \
#   > logs_test/$model_name'_'M_ETTh2_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# # ETTm1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_96_$preLen \
#   --model Autoformer \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --test_train_num $test_train_num \
#   --gpu 2 \
#   > logs_test/$model_name'_'M_ETTm1_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_$preLen \
  --model Autoformer \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --test_train_num $test_train_num \
  --gpu 2 \
  > logs_test/$model_name'_'M_ETTm2_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log


done