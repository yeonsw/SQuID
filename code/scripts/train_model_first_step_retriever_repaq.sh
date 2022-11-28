DATA_DIR=$WORKING_DIR/data/paq
OUTPUT_DIR=$WORKING_DIR/outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

TRAIN_FILE=$OUTPUT_DIR/nq/nq_train-train_dpr_result.jsonl
EVAL_FILE=$OUTPUT_DIR/nq/nq_train-dev_dpr_result.jsonl
PRED_FILE=$OUTPUT_DIR/nq/nq_test_dpr_result.jsonl

N_HARD_NEGS=16
F1_SIM_LMB=0.5
DPR_CHECKPOINT=$CHECKPOINT_DIR/squid/squid_nq_repaq
INIT_CHECKPOINT=facebook/dpr-question_encoder-single-nq-base

python train_model.py \
    --train_file $TRAIN_FILE \
    --eval_file $EVAL_FILE \
    --max_length 128 \
    --n_hard_negs $N_HARD_NEGS \
    --init_checkpoint $INIT_CHECKPOINT \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --per_device_inds_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_steps 800 \
    --n_epochs 3 \
    --q_sim_lmb 0.0 \
    --f1_sim_lmb $F1_SIM_LMB \
    --exact_sim_lmb 0.0 \
    --learning_rate 1e-5 \
    --checkpoint_save_dir $DPR_CHECKPOINT
