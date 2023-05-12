
model_name=Autoformer

test_train_num=10

gpu=2


for preLen in 96
do

# Traffic
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_$preLen \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3
  --test_train_num $test_train_num \
  --gpu $gpu \
  > logs_test/$model_name'_'M_Traffic_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done