import os

import argparse
import collections
import csv
import json
import jsonlines
import random
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--question_file", type=str, default=None)
    parser.add_argument("--qas_pyserini_input_format_file", type=str, default=None)

    args = parser.parse_args()
    return args

def format_qas2pyserini(question_file, qas_pyserini_input_file):
    os.makedirs(os.path.dirname(qas_pyserini_input_file), exist_ok=True)
    with open(qas_pyserini_input_file, "w") as fw:
        writer = csv.writer(fw, delimiter="\t")
        with jsonlines.open(question_file, "r") as reader:
            for idx, inst in enumerate(tqdm(reader, desc="QAs -> Pyserini format")):
                writer.writerow([idx, inst["question"]])
    return 0

def main(args):
    _ = format_qas2pyserini(args.question_file, args.qas_pyserini_input_format_file)
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
