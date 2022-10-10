import os
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain
import argparse
from tqdm import tqdm
from logdeep.dataset.session import sliding_window

# logbert/BGL/logbert.py

import os
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from logdeep.tools.utils import *

#os.makedirs("./content/logbert/output/bgl/bert", exist_ok= True) #!mkdir -p ./logbert/output/bgl/bert

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

options["output_dir"] = "./content/logbert/output/bgl/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options['output_dir'] + 'train'
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10

options["mask_ratio"] = 0.5

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 0
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 15
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])


Trainer(options).train()