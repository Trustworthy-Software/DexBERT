export DATA_FILE=../Data/data/pretrain_data/data_file.txt
export VOCAB_FILE=../Data/data/pretrain_data/vocab.txt
export SAVE_DIR=../save_dir/DexBERT
export LOG_DIR=../log_dir/DexBERT

python pretrainDexBERT.py \
    --train_cfg config/DexBERT/pretrain.json \
    --model_cfg config/DexBERT/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $VOCAB_FILE \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
