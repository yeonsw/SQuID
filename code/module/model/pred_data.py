#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dataclasses import dataclass, field
import logging
import random
import json
import jsonlines
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
from typing import Optional

from .tokenizer import get_tokenizer
from .metric import compute_f1_sets, compute_exact_sets

logger = logging.getLogger(__name__)

@dataclass
class DataPredArguments:
    pred_file: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=128)

    def __post_init__(self):
        if self.pred_file is None:
            raise ValueError("Need either a training/evalation file.")

class DPRPredDataset(Dataset):
    def __init__(self, input_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.questions, \
            self.questions_tok, \
            self.answers, \
            self.bm25_questions, \
            self.bm25_questions_tok, \
            self.bm25_answers, \
            self.bm25_sims \
                = self.get_data(input_file)
        assert len(self.questions_tok["input_ids"]) == len(self.answers)
    
    def get_data(self, fname):
        with jsonlines.open(fname, "r") as reader:
            qas = [ \
                r for r in tqdm( \
                            reader, desc="Reading {}".format(fname.split('/')[-1])) \
            ]
        
        questions = []
        answers = []
        bm25_questions = []
        bm25_answers = []
        bm25_sims = []
        for qa in tqdm(qas, desc="Building dataset"):
            questions.append(qa["question"])
            answers.append(qa["answer"])
            bm25_qs = []
            bm25_ans = []
            bm25_sim = []
            for rqa in qa["retrieved_qas"][:100]:
                bm25_qs.append(rqa["question"])
                bm25_ans.append(rqa["answer"])
                bm25_sim.append(rqa["sim"])
            bm25_questions.append(bm25_qs)
            bm25_answers.append(bm25_ans)
            bm25_sims.append(bm25_sim)
        
        print("N Qs, N BM 25 Qs | {} {}".format(len(questions), len(bm25_questions)))
        print("Tokenizing questions")
        questions_tok = \
            self.tokenizer.batch_encode_plus( \
                questions, \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            )
        print("Tokenizing BM25 questions")
        bm25_questions_tok = [ \
            self.tokenizer.batch_encode_plus( \
                bm25_qs, \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            ) for bm25_qs in tqdm(bm25_questions, desc="Tokenizing BM25 questions") \
        ]
        return ( \
            questions,
            questions_tok, \
            answers, \
            bm25_questions,
            bm25_questions_tok, \
            bm25_answers, \
            bm25_sims,
        )
    
    def __len__(self):
        return len(self.questions_tok["input_ids"])

    def __getitem__(self, idx):
        question = self.questions[idx]
        q_input_ids = self.questions_tok["input_ids"][idx]
        q_token_type_ids = self.questions_tok["token_type_ids"][idx]
        q_attention_mask = self.questions_tok["attention_mask"][idx]
        q_answer = self.answers[idx]
        
        bm25_questions = self.bm25_questions[idx]
        bm25_input_ids = self.bm25_questions_tok[idx]["input_ids"]
        bm25_token_type_ids = self.bm25_questions_tok[idx]["token_type_ids"]
        bm25_attention_mask = self.bm25_questions_tok[idx]["attention_mask"]
        bm25_answer = self.bm25_answers[idx]
        bm25_sim = self.bm25_sims[idx]

        bm25_size = len(bm25_answer)

        return {
            "question": question,
            "q_input_ids": q_input_ids,
            "q_token_type_ids": q_token_type_ids,
            "q_attention_mask": q_attention_mask,
            "bm25_questions": bm25_questions,
            "bm25_input_ids": bm25_input_ids,
            "bm25_token_type_ids": bm25_token_type_ids,
            "bm25_attention_mask": bm25_attention_mask,
            "q_answer": q_answer,
            "bm25_answer": bm25_answer,
            "bm25_size": bm25_size,
            "bm25_sim": bm25_sim
        }

def pred_data_collator(samples):
    if len(samples) == 0:
        return {}
    bsize = len(samples)
    bm25_size = [s["bm25_size"] for s in samples]
    assert all([bm25_size[0] == s for s in bm25_size])
    
    questions = [s["question"] for s in samples]
    q_input_ids = torch.stack([s["q_input_ids"] for s in samples])
    q_token_type_ids = torch.stack([s["q_token_type_ids"] for s in samples])
    q_attention_mask = torch.stack([s["q_attention_mask"] for s in samples])
    
    bm25_questions = [s["bm25_questions"] for s in samples]
    bm25_input_ids = torch.cat([s["bm25_input_ids"] for s in samples], dim=0)
    bm25_token_type_ids = torch.cat([s["bm25_token_type_ids"] for s in samples], dim=0)
    bm25_attention_mask = torch.cat([s["bm25_attention_mask"] for s in samples], dim=0)
    
    input_ids = torch.cat((q_input_ids, bm25_input_ids))
    token_type_ids = torch.cat((q_token_type_ids, bm25_token_type_ids))
    attention_mask = torch.cat((q_attention_mask, bm25_attention_mask))
    bm25_answers = [ \
        s["bm25_answer"] for s in samples \
    ]
    bm25_sim = [ \
        s["bm25_sim"] for s in samples \
    ]
    q_answer = [s["q_answer"] for s in samples]
    return {
        "n_samples": bsize,
        "bm25_size": bm25_size,
        "questions": questions,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "bm25_questions": bm25_questions,
        "bm25_answers": bm25_answers,
        "bm25_sim": bm25_sim,
        "gt_answer": q_answer
    }
