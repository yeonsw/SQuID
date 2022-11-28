import collections
import csv
import json
import jsonlines
import os
import random
import numpy as np
from tqdm import tqdm
from .paq.retrievers import retrieve

class DPRRetriever:
    def __init__(self, args):
        self.args = args
   
    def search(self):
        assert self.args.model != None \
            and self.args.paq_qas_file != None \
            and self.args.qas_file != None \
            and self.args.topk != None \
            and self.args.faiss_index_file != None \
            and self.args.dpr_retrieval_result != None

        data = retrieve.main( \
            self.args.model, \
            self.args.qas_file, \
            self.args.paq_qas_file, \
            self.args.topk, \
            self.args.faiss_index_file, \
            fp16=True, \
            memory_friendly_parsing=True, \
            verbose=True, \
            batch_size=512
        )

        os.makedirs(os.path.dirname(self.args.dpr_retrieval_result), exist_ok=True)
        with jsonlines.open(self.args.dpr_retrieval_result, "w") as writer:
            for d in data:
                retrieved_qas = []
                for rqa in d["retrieved_qas"]:
                    retrieved_qas.append({
                        "question": rqa["question"],
                        "answer": rqa["answer"],
                        "sim": -rqa["score"]
                    })
                retrieved_qas.sort(key=lambda x: x["sim"], reverse=True)
                instance = {
                    "question": d["input_qa"]["question"],
                    "answer": d["input_qa"]["answer"],
                    "retrieved_qas": retrieved_qas
                }
                writer.write(instance)
        return 0

