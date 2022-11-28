import os

import argparse
import collections
import csv
import json
import jsonlines
import random
import numpy as np
from tqdm import tqdm

def read_paq_file(paq_file):
    paq = []
    with jsonlines.open(paq_file, "r") as reader:
        for i, inst in enumerate(tqdm(reader, desc="Reading the PAQ file")):
            paq.append({
                "question": inst["question"],
                "answer": inst["answer"]
            })
    return paq

def read_qfile(question_file):
    queries = []
    with jsonlines.open(question_file, "r") as reader:
        for idx, qa in enumerate(tqdm(reader, desc="Reading the ODQA file")):
            queries.append({
                "id": idx,
                "question": qa["question"], \
                "answer": qa["answer"]
            })
    return queries

def read_retrieval_result_file(retrieval_result_file):
    data = collections.defaultdict(list)
    with open(retrieval_result_file, "r") as fr:
        for line in tqdm(fr.readlines(), desc="Reading the retrieval result file"):
            qid, _, rqid, rank, sim, _ = line.split()
            data[int(qid)].append({
                "rank": int(rank),
                "rqid": int(rqid),
                "sim": float(sim)
            })
        for qid in data:
            data[qid].sort(key=lambda x: x["rank"])
    
    qid2rqs = {}
    for qid in data:
        bm25_docs = [ \
            (inst["rqid"], inst["sim"]) \
                for inst in data[qid] \
        ]
        qid2rqs[qid] = bm25_docs
    return qid2rqs

def format_pyserini_retrieval_result2squid_input(args):
    
    qid2results = \
        read_retrieval_result_file( \
            args.pyserini_output_file \
        )
    
    paq = read_paq_file( \
        args.paq_file, \
    )
    qas = read_qfile(args.qas_file)

    with jsonlines.open(args.qas_bm25_retrieval_result_file, "w") as writer:
        for qa in tqdm(qas, desc="QAs Pyserini output -> SQUID input format"):
            #qa["answer"]: list of str
            rqas = []
            for rqid, sim in qid2results[qa["id"]]:
                target_paq_qa = paq[rqid]
                rqas.append({
                    "question": target_paq_qa["question"],
                    "answer": target_paq_qa["answer"],
                    "sim": sim
                })
            
            writer.write({
                "question": qa["question"],
                "answer": qa["answer"],
                "retrieved_qas": rqas
            })
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--paq_file", type=str, default=None)
    parser.add_argument("--pyserini_output_file", type=str, default=None)
    parser.add_argument("--qas_file", type=str, default=None)
    parser.add_argument("--qas_bm25_retrieval_result_file", type=str, default=None)
    
    args = parser.parse_args()
    return args

def main(args):
    _ = format_pyserini_retrieval_result2squid_input(args)
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
