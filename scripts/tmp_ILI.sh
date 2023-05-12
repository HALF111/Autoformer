
model_name=Autoformer

test_train_num=10

gpu=2


for preLen in 24

# ILI
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_$preLen \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
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
  --gpu $gpu \
  > logs_test/$model_name'_'M_ILI_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done
