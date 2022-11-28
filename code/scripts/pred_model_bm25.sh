DATA_DIR=$WORKING_DIR/data/paq
OUTPUT_DIR=$WORKING_DIR/outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

PRED_FILE=$OUTPUT_DIR/nq/nq_test_bm25_result.jsonl
DPR_CHECKPOINT=$CHECKPOINT_DIR/squid/squid_nq_bm25

python pred_model.py \
    --pred_file $PRED_FILE \
    --max_length 128 \
    --init_checkpoint $DPR_CHECKPOINT \
    --per_device_eval_batch_size 2 \
