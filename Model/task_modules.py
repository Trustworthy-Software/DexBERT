import json
import torch
import torch.nn as nn
from typing import NamedTuple

import models

from pretrainDexBERT import Pipeline
from utils import truncate_tokens

class Config(NamedTuple):
    "Configuration for prediction model & training"
    # model
    input_len: int = 128
    embedding_len: int = 64
    key_instance_num: int = 1

    # training
    Bert_batch_size: int = 32
    MIL_batch_size: int  = 1
    MCD_batch_size: int  = 768
    lr: float = 1e-4
    warmup: float = 0.1
    n_epochs: int =25
    Adam_Betas: tuple = (0.9, 0.999)
    Adam_WeightDecay: float = 10e-5
    save_steps: int = 10000
    total_steps: int = 1000000

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

class ClassSeqDataLoader():
    """ Load class sequence from a pre-processed APK txt file. 
    """
    def __init__(self, file, malicious_classes, batch_size, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.file = open(file, "r", encoding='utf-8', errors='ignore') 
        self.malicious_classes = {}
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.current_class_id = 0
        self.current_class_name = ''

        for class_name in malicious_classes:
            if class_name.startswith('L'):
                self.malicious_classes[class_name[1:]] = ''
            else:
                self.malicious_classes[class_name] = ''

    def read_tokens(self, f, length, discard_last_and_restart=False, keep_method_name=True):
        """ Read tokens from file pointer with limited length """
        tokens   = []
        ClassEnd = False
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None, ClassEnd
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens, ClassEnd # return last tokens in the document
            if line.strip().startswith('ClassName:'):
                class_name = line.strip().split(' ')[-1]
                if class_name.startswith('L'):
                    self.current_class_name = class_name[1:].split('$')[0]
                else:
                    self.current_class_name = class_name.split('$')[0]
                continue  # skip the smali class name
            if line.strip().startswith('MethodName:') and not keep_method_name:
                continue # skip the smali method name
            if line.strip().startswith('ClassEnd'):
                ClassEnd = True
                return tokens, ClassEnd
            tokens.extend(self.tokenize(line.strip()))
        return tokens, ClassEnd
    
    def __iter__(self): # iterator to load data
        close_file = False
        while True and not close_file:
            batch = []
            batch_class_names = []
            for _ in range(self.batch_size):
                len_tokens = self.max_len

                tokens, ClassEnd = self.read_tokens(self.file, len_tokens, discard_last_and_restart=False, keep_method_name=True)
                
                if ClassEnd:  # end of current class -> end of current batch
                    self.current_class_id += 1
                
                if tokens is None:  # end of file
                    self.file.close()
                    close_file = True
                    break
                if len(tokens) == 0:
                    continue 

                class_id = self.current_class_id
                if self.current_class_name in self.malicious_classes:
                    class_label = 1
                else:
                    class_label = 0
                instance = (tokens, class_id, class_label)
                for proc in self.pipeline:
                    instance = proc(instance)
                batch.append(instance)
                batch_class_names.append(self.current_class_name)

            if len(batch) == 0:
                continue
            # To Tensor
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors, batch_class_names


class Preprocess4EmbeddingIntegration(Pipeline):
    """ Pre-processing steps for embedding integration. """
    def __init__(self, indexer, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer # function from token to token index
    
    def __call__(self, instance: tuple):
        tokens, class_id, class_label = instance

        truncate_tokens(tokens, self.max_len)

        segment_ids = [0]*len(tokens)
        input_mask  = [1]*len(tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)

         # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
    
        return (input_ids, segment_ids, input_mask, class_id, class_label)

class BertAEModel(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)

        # auto-encoder
        self.AE_Layer_1 = nn.Linear(cfg.dim, cfg.max_len)
        self.AE_Layer_2 = nn.Linear(cfg.max_len, cfg.class_vec_len)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)

        # auto-encoder
        # r1 = torch.flatten(h, start_dim=1)
        r1 = h[:,0]
        x = self.AE_Layer_1(r1)
        r2 = self.AE_Layer_2(x)

        return r2

class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        prediction = self.classifier(x)
        return prediction