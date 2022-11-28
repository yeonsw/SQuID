DATA_DIR=$WORKING_DIR/data/paq
OUTPUT_DIR=$WORKING_DIR/outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

TARGETS="train-train train-dev test"
for TARGET in $TARGETS
do
    QAS_FILE=$DATA_DIR/nq/NQ-open.$TARGET.jsonl
    QAS_PYSERINI_INPUT_FILE=$OUTPUT_DIR/nq/nq_$TARGET\_pyserini_input_format.tsv
    python qas2pyserini_input_format.py \
        --question_file $QAS_FILE \
        --qas_pyserini_input_format_file $QAS_PYSERINI_INPUT_FILE
    
    mkdir -p $OUTPUT_DIR/nq
    PAQ_PYSERINI_INDEX_FILE=$OUTPUT_DIR/paq_outputs/tqa-train-nq-train-PAQ_pyserini_index
    PYSERINI_SEARCH_RESULT_FILE=$OUTPUT_DIR/nq/nq_$TARGET\_bm25_result.txt
    python -m pyserini.search \
        --topics $QAS_PYSERINI_INPUT_FILE \
        --index $PAQ_PYSERINI_INDEX_FILE \
        --output $PYSERINI_SEARCH_RESULT_FILE \
        --bm25 \
        --hits 50 \
        --batch-size 1024 \
        --threads 32

    PAQ_FILE=$DATA_DIR/paq_qas/tqa-train-nq-train-PAQ.jsonl
    QAS_BM25=$OUTPUT_DIR/nq/nq_$TARGET\_bm25_result.jsonl
    python qas_pyserini_output2squid_input.py \
        --paq_file $PAQ_FILE \
        --pyserini_output_file $PYSERINI_SEARCH_RESULT_FILE \
        --qas_file $QAS_FILE \
        --qas_bm25_retrieval_result_file $QAS_BM25
done
