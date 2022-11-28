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
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=128)
    n_hard_negs: Optional[int] = field(default=8)
    q_sim_lmb: Optional[float] = field(default=0.5)
    f1_sim_lmb: Optional[float] = field(default=0.5)
    exact_sim_lmb: Optional[float] = field(default=0.0)

    def __post_init__(self):
        if self.train_file is None or self.eval_file is None:
            raise ValueError("Need either a training/evalation file.")

class DPRDataset(Dataset):
    def __init__(self, input_file, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length \
            = self.config.max_length
        self.questions_tokenized = \
            self.get_data(input_file)
        
        self.pos_inds = None
        self.neg_inds = None
        scores = []
        for qa in self.questions_tokenized:
            scores.append([rqa["sim"] for rqa in qa["retrieved_qas"]])
        _ = self.update_inds(scores) 
    
    def compute_syn_score(self, x, answer):
        return compute_f1_sets(x["answer"], answer)
    
    def update_inds(self, all_scores):
        n = 50
        err = 0
        pos_inds = []
        neg_inds = []
        for i, scores in enumerate(tqdm(all_scores, desc="Updating inds")):
            target_data = self.questions_tokenized[i]
            question = target_data["question"]
            answer = target_data["answer"]
            rqas = target_data["retrieved_qas"]
            retrieved = [(j, s, rqa) for j, s, rqa in zip(range(len(rqas)), scores, rqas)]
            retrieved = sorted( \
                retrieved, key=lambda x: x[1], reverse=True \
            )
            retrieved = sorted( \
                retrieved, \
                key=lambda x: self.compute_syn_score( \
                    x[2], answer, \
                ), \
                reverse=True \
            )
            retrieved = [rqa for rqa in retrieved if rqa[2]["question"] != question][:n]
        
            pos_inds.append(retrieved[0][0])
            neg_ind = [
                rqa[0] for rqa in retrieved[1:] \
                    if compute_exact_sets(rqa[2]["answer"], answer) == 0 \
            ]
            if len(neg_ind) == 0:
                err += 1
            neg_inds.append(neg_ind)
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        print("Error: {}({}/{})".format( \
            err/len(all_scores), err, len(all_scores) \
        ))
        return 0
        
    def get_data(self, fname):
        with jsonlines.open(fname, "r") as reader:
            qas = [ \
                r for r in tqdm( \
                    reader, desc="Reading {}".format(fname.split('/')[-1])) \
            ]
        print("Tokenizing questions")
        questions_tokenized = \
            self.tokenizer.batch_encode_plus( \
                [qa["question"] for qa in qas], \
                padding="max_length", \
                max_length=self.max_length, \
                truncation=True, \
                return_tensors='pt' \
            )
        questions_tokenized = \
            [{"question": qa["question"], "answer": qa["answer"], "input_ids": iid, "token_type_ids": tid, "attention_mask": am, "retrieved_qas": []} \
                for iid, tid, am, qa in zip( \
                    questions_tokenized["input_ids"], \
                    questions_tokenized["token_type_ids"], \
                    questions_tokenized["attention_mask"], \
                    qas, \
                ) \
            ]
        print("Tokenizing retrieved questions")
        for qt, qa in tqdm(zip(questions_tokenized, qas), total=len(qas)):
            retrieved_qs = [_["question"] for _ in qa["retrieved_qas"]]
            retrieved_qs_tokenized = \
                self.tokenizer.batch_encode_plus( \
                    retrieved_qs, \
                    padding="max_length", \
                    max_length=self.max_length, \
                    truncation=True, \
                    return_tensors='pt' \
                )
            qt["retrieved_qas"] = [ \
                { \
                    "question": rqa["question"], \
                    "answer": rqa["answer"], \
                    "input_ids": iid, \
                    "token_type_ids": tid, \
                    "attention_mask": am, \
                    "sim": rqa["sim"] \
                } for iid, tid, am, rqa in zip( \
                    retrieved_qs_tokenized["input_ids"], \
                    retrieved_qs_tokenized["token_type_ids"], \
                    retrieved_qs_tokenized["attention_mask"], \
                    qa["retrieved_qas"], \
                ) \
            ]
        
        return questions_tokenized
    
    def __len__(self):
        return len(self.questions_tokenized)

    def __getitem__(self, idx):
        example = self.questions_tokenized[idx]
        positive_ind = self.pos_inds[idx]
        positive_example = example["retrieved_qas"][positive_ind]
        
        negative_inds = self.neg_inds[idx]
        if len(negative_inds) != 0:
            negative_inds = random.sample( \
                negative_inds, \
                min(self.config.n_hard_negs, len(negative_inds)) \
            )
        negative_examples = []
        if len(negative_inds) != 0: 
            negative_examples = [example["retrieved_qas"][i] for i in negative_inds]
        
        return {
            "example": example,
            "positive_example": positive_example,
            "negative_examples": negative_examples,
        }

def data_collator(samples):
    if len(samples) == 0:
        return {}
    bsize = len(samples)    
    
    q_input_ids = torch.stack([s["example"]["input_ids"] for s in samples])
    q_token_type_ids = torch.stack([s["example"]["token_type_ids"] for s in samples])
    q_attention_mask = torch.stack([s["example"]["attention_mask"] for s in samples])
    
    pos_input_ids = torch.stack([s["positive_example"]["input_ids"] for s in samples])
    pos_token_type_ids = torch.stack([s["positive_example"]["token_type_ids"] for s in samples])
    pos_attention_mask = torch.stack([s["positive_example"]["attention_mask"] for s in samples])
    
    neg_input_ids, neg_token_type_ids, neg_attention_mask = [], [], []
    for s in samples:
        for neg in s["negative_examples"]:
            neg_input_ids.append(neg["input_ids"])
            neg_token_type_ids.append(neg["token_type_ids"])
            neg_attention_mask.append(neg["attention_mask"])
    if len(neg_input_ids) != 0:
        neg_input_ids = torch.stack(neg_input_ids, dim=0)
        neg_token_type_ids = torch.stack(neg_token_type_ids, dim=0)
        neg_attention_mask = torch.stack(neg_attention_mask, dim=0)
    else:
        neg_input_ids, neg_token_type_ids, neg_attention_mask = (torch.tensor([]), torch.tensor([]), torch.tensor([]))
    
    input_ids = torch.cat((q_input_ids, pos_input_ids, neg_input_ids))
    token_type_ids = torch.cat((q_token_type_ids, pos_token_type_ids, neg_token_type_ids))
    attention_mask = torch.cat((q_attention_mask, pos_attention_mask, neg_attention_mask))

    answers = [s["example"]["answer"] for s in samples] \
        + [s["positive_example"]["answer"] for s in samples] \
        + [neg["answer"] for s in samples for neg in s["negative_examples"]]
    q_answer = [s["example"]["answer"] for s in samples]
    return {
        "n_samples": bsize,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "answers": answers,
        "gt_answer": q_answer
    }
