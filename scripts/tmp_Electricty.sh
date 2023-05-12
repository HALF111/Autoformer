
model_name=Autoformer

test_train_num=10

gpu=2


for preLen in 96
do

# Electricty
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_$preLen \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --test_train_num $test_train_num \
  --gpu $gpu \
  > logs_test/$model_name'_'M_Electricty_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done