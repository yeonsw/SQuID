# Two-Step Question Retrieval for Open-Domain QA

Yeon Seonwoo\*, Juhee Son\*, Jiho Jin, Sang-Woo Lee, Ji-Hoon Kim, Jung-Woo Ha, and Alice Oh (\*equal contribution) | Findings of ACL 2022 | [Paper](https://aclanthology.org/2022.findings-acl.117/) 

KAIST, Naver CLOVA, Naver AI Lab

Official implementation of **Two-Step Question Retrieval for Open-Domain QA**

# Abstract

The retriever-reader pipeline has shown promising performance in open-domain QA but suffers from a very slow inference speed. Recently proposed question retrieval models tackle this problem by indexing question-answer pairs and searching for similar questions. These models have shown a significant increase in inference speed, but at the cost of lower QA performance compared to the retriever-reader models. This paper proposes a two-step question retrieval model, SQuID (Sequential QuestionIndexed Dense retrieval) and distant supervision for training. SQuID uses two bi-encoders for question retrieval. The first-step retriever selects top-k similar questions, and the secondstep retriever finds the most similar question from the top-k questions. We evaluate the performance and the computational efficiency of SQuID. The results show that SQuID significantly increases the performance of existing question retrieval models with a negligible loss on inference speed.

# Getting started

## Requirements

1. Java 11 (to run Pyserini)
2. Nvidia [apex](https://github.com/NVIDIA/apex)
3. Python libraries in requirements.txt

## Setup

1. Set the project directory

```
export WORKING_DIR=/path/to/this/project/folder
```

2. Download PAQ files

```
cd $WORKING_DIR
wget https://dl.fbaipublicfiles.com/paq/v1/TQA_TRAIN_NQ_TRAIN_PAQ.tar.gz
tar -xvzf TQA_TRAIN_NQ_TRAIN_PAQ.tar.gz
mkdir -p $WORKING_DIR/data/paq/paq_qas/
mv TQA_TRAIN_NQ_TRAIN_PAQ/tqa_train_nq_train_PAQ.jsonl $WORKING_DIR/data/paq/paq_qas/
```

3. Download the PAQ retriever

```
cd $WORKING_DIR
wget https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_multi_base_256.tar.gz
tar -xvzf retriever_multi_base_256.tar.gz
mkdir -p checkpoints/paq_dpr
mv retriever_multi_base_256 checkpoints/paq_dpr/
```

4. Download the PAQ faiss index

```
cd $WORKING_DIR
wget https://dl.fbaipublicfiles.com/paq/v1/models/indices/multi_base_256.hnsw.sq8.faiss
mv multi_base_256.hnsw.sq8.faiss data/paq/paq_qas/
```

5. Download PAQ code
```
cd $WORKING_DIR/code/module/retriever/
git clone https://github.com/facebookresearch/PAQ.git
mv PAQ/paq ./
```

6. Download NQ files

```
mkdir -p $WORKING_DIR/data/paq/nq
cd $WORKING_DIR/data/paq/nq/
wget https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-train.jsonl
wget https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-dev.jsonl
wget https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.test.jsonl
```

## Training and evaluation

### First-step retrieval

1. Option 1: BM25 as the first-step retriever

This script generates an index file for BM25 retrieval and runs BM25

```
bash scripts/_0_paq2index.sh
bash scripts/_1_bm25_search.sh
```

2. Option 2: RePAQ as the first-step retriever

This script runs RePAQ retrieval on NQ-train/dev/test sets

```
bash scripts/_1_dpr_search.sh
```


### Training SQuID

1. Option 1: BM25

```
bash scripts/_2_train_model_first_step_retriever_bm25.sh
```

2. Option 2: RePAQ

```
bash scripts/_2_train_model_first_step_retriever_repaq.sh
```

### Inference

1. Option 1: BM25

```
bash scripts/_3_pred_model_bm25.sh
```

2. Option 2: RePAQ

```
bash scripts/_3_pred_model_dpr.sh

```

# Code reference

https://github.com/facebookresearch/PAQ

# Copyright

Copyright 2022-present NAVER Corp. and KAIST (Korea Advanced Institute of Science and Technology)

# Acknowledgement

This work was partly supported by NAVER Corp. and the Engineering Research Center Program through the National Research Foundation of Korea (NRF) funded by the Korean Government MSIT (NRF-2018R1A5A1059921).
