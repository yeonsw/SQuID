import os

import argparse
import jsonlines
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--paq_file", type=str, default=None)
    parser.add_argument("--paq_pyserini_input_format_file", type=str, default=None)
    parser.add_argument("--paq_pyserini_index_file", type=str, default=None)

    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(os.path.dirname(args.paq_pyserini_input_format_file), exist_ok=True)
    with jsonlines.open(args.paq_pyserini_input_format_file, "w") as writer:
        with jsonlines.open(args.paq_file, "r") as reader:
            for i, inst in enumerate( \
                tqdm(reader, desc="PAQ -> Pyserini input format") \
            ):
                writer.write({
                    "id": str(i),
                    "contents": inst["question"],
                })
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
