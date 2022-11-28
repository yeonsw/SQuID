import argparse
from dataclasses import dataclass, field
import random
import numpy as np
from transformers import HfArgumentParser
import torch
from typing import Optional

from module.model.model import PredModelArguments, DPRQsimPred
from module.model.pred_data import DataPredArguments

@dataclass
class Additional:
    seed: Optional[int] = field(default=42)

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return 0

def main():
    parser = HfArgumentParser((DataPredArguments, PredModelArguments, Additional))
    data_args, model_args, additional_args = parser.parse_args_into_dataclasses()
    
    _ = set_seed(additional_args.seed)
    
    model = DPRQsimPred(model_args, data_args)
    model.pred_model()
    return 0

if __name__ == "__main__":
    _ = main()

