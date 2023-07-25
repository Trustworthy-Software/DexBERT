import os
import os.path as osp
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import DownloadApk, Disassemble, get_device
from SmaliPreprocess import Smalis2Txt

import tokenization
from models import DexBERT, Config
from dataloader import PreprocessEmbedding


class SmaliSeqDataset(Dataset):
    def __init__(self, file, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.file = open(file, "r", encoding='utf-8', errors='ignore') 
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.current_class_id = 0
        
        self.instance_list = self.instance_generator()

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
                continue  # skip the smali class name
            if line.strip().startswith('MethodName:') and not keep_method_name:
                continue # skip the smali method name
            if line.strip().startswith('ClassEnd'):
                ClassEnd = True
                return tokens, ClassEnd
            tokens.extend(self.tokenize(line.strip()))
        return tokens, ClassEnd

    def instance_generator(self): # iterator to load data
        instance_list = []
        close_file = False
        while True and not close_file:
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
            instance = (tokens, class_id)
            for proc in self.pipeline:
                instance = proc(instance)
            
            instance_list.append(instance)
        return instance_list
    
    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        input_ids, segment_ids, input_mask, class_id = self.instance_list[index]
        return np.array(input_ids), np.array(segment_ids), np.array(input_mask), np.array(class_id)


def BertInfer(BertAEmodel, dataloader, device):
    class_vector_list = []
    last_class_id  = 0

    seq_iter_bar = tqdm(dataloader)
    with torch.no_grad():
        for _, batch in enumerate(seq_iter_bar):
            batch = [t.to(device) for t in batch]
            input_ids, segment_ids, input_mask, class_id = batch

            r2 = BertAEmodel(input_ids, segment_ids, input_mask)
            batch_vec = r2.cpu().detach().numpy()
            
            for i, emb in enumerate(batch_vec):
                if len(class_vector_list) == 0:
                    class_vector_list.append(np.expand_dims(emb, axis=0))
                    continue
                if int(class_id.cpu()[i]) == last_class_id:
                    class_vector_list[-1] = np.concatenate([class_vector_list[-1], np.expand_dims(emb, axis=0)])
                    continue
                class_vector_list.append(np.expand_dims(emb, axis=0))
                last_class_id = int(class_id.cpu()[i])
    
    return class_vector_list

def Hash2ApkEmb(hash, tmp_dir, save_dir, BertAE, batch_size, pipeline, only_api_instruction, device):
    apk_path = osp.join(tmp_dir, hash.upper()+'.apk')
    DownloadApk(apk_path)
    smali_dir = osp.join(tmp_dir, hash)
    Disassemble(apk_path, smali_dir)
    Smalis2Txt(tmp_dir, smali_dir, only_keep_func_name=only_api_instruction)
    ApkName = smali_dir.split('/')[-1] if smali_dir.split('/')[-1] else smali_dir.split('/')[-2]
    txt_file = osp.join(tmp_dir, ApkName+'.txt')

    dataset = SmaliSeqDataset(txt_file, tokenize, Bert_model_cfg.max_len, pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_vec_list = BertInfer(BertAE, dataloader, device)

    with open(osp.join(save_dir, ApkName+'.pkl'), 'wb') as f:
        pickle.dump(class_vec_list, f)
    os.system('rm -r {}'.format(osp.join(tmp_dir, '*')))

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    
    root_dir = './data'
    src_data_list = [['hash_list.txt', 'embeddings']] # os.path.join(root_dir, 'hash_list.txt') is the hash list of your APKs, os.path.join(root_dir, 'embeddings') is the folder where you will save your APK embeddings

    Bert_model_cfg = './bert_base.json'
    vocab = './vocab.txt'
    DexBERT_file = './pretrained_dexbert_model_steps_604364.pt'  # please download with link: https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view?usp=sharing

    only_api_instruction = False

    # model initialization
    batch_size = 32
    device = get_device()
    
    Bert_model_cfg = Config.from_json(Bert_model_cfg)
    BertAE = DexBERT(Bert_model_cfg)
    BertAE.load_state_dict(torch.load(DexBERT_file), strict=False)
    BertAE.to(device)
    BertAE.eval()
    
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))  

    pipeline = [PreprocessEmbedding(tokenizer.convert_tokens_to_ids)]

    for pair in src_data_list:
        src_path, data_dir = pair[0], pair[1]
        hash_list = open(osp.join(root_dir, src_path), 'r').readlines()
        save_dir = osp.join(root_dir, data_dir)
        tmp_dir = osp.join(save_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        for hash in tqdm(hash_list): 
            hash = hash.strip()
            if os.path.exists(os.path.join(save_dir, hash.upper()+'.pkl')):
                continue
            Hash2ApkEmb(hash, tmp_dir, save_dir, BertAE, batch_size, pipeline, only_api_instruction, device)
    os.system('rm -r {}'.format(tmp_dir))

            