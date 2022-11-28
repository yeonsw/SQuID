import argparse

from module.retriever import dpr

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--paq_qas_file", type=str, default=None)
    parser.add_argument("--faiss_index_file", type=str, default=None)
    parser.add_argument("--qas_file", type=str, default=None)
    parser.add_argument("--dpr_retrieval_result", type=str, default=None)
    
    parser.add_argument("--topk", type=int, default=50)

    args = parser.parse_args()
    return args

def main(args):
    retriever = dpr.DPRRetriever(args)
    retriever.search()
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
