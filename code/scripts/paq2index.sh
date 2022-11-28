DATA_DIR=$WORKING_DIR/data/paq
OUTPUT_DIR=$WORKING_DIR/outputs/paq_outputs
CHECKPOINT_DIR=$WORKING_DIR/checkpoints

PAQ_FILE=$DATA_DIR/paq_qas/tqa-train-nq-train-PAQ.jsonl
PAQ_PYSERINI_INDEX_FILE=$OUTPUT_DIR/tqa-train-nq-train-PAQ_pyserini_index

PAQ_PYSERINI_INPUT_DIR=$OUTPUT_DIR/paq_pyserini_format
PAQ_PYSERINI_INPUT_FILE=$PAQ_PYSERINI_INPUT_DIR/paq_pyserini_file.jsonl

python paq2pyserini_format.py \
    --paq_file $PAQ_FILE \
    --paq_pyserini_input_format_file $PAQ_PYSERINI_INPUT_FILE \
   
python -m pyserini.index \
    --collection JsonCollection \
    --generator DefaultLuceneDocumentGenerator \
    --threads 10 \
    --input $PAQ_PYSERINI_INPUT_DIR \
    --index $PAQ_PYSERINI_INDEX_FILE \
    --storePositions \
    --storeDocvectors \
    --storeRaw

