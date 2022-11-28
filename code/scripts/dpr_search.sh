DATA_DIR=$WORKING_DIR/data/paq
OUTPUT_DIR=$WORKING_DIR/outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

MODEL=$CHECKPOINT_DIR/paq_dpr/retriever_multi_base_256
PAQ_QAS_FILE=$DATA_DIR/paq_qas/tqa-train-nq-train-PAQ.jsonl
FAISS_INDEX=$DATA_DIR/paq_qas/multi_base_256.hnsw.sq8.faiss
TOPK=50

TARGETS="train-train train-dev test"
for TARGET in $TARGETS
do
    QAS_FILE=$DATA_DIR/nq/NQ-open.$TARGET.jsonl
    DPR_RETRIEVAL_RESULT=$OUTPUT_DIR/nq/nq_$TARGET\_dpr_result.jsonl
    python dpr_search.py \
        --model $MODEL \
        --paq_qas_file $PAQ_QAS_FILE \
        --faiss_index_file $FAISS_INDEX \
        --qas_file $QAS_FILE \
        --dpr_retrieval_result $DPR_RETRIEVAL_RESULT \
        --topk $TOPK
done
