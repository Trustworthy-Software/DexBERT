import models
import tokenization
import torch
import fire
import numpy as np

from pretrainDexBERT import Preprocess4Pretrain, SentPairDataLoader, BertAEModel4Pretrain
from colorama import Fore, Style
from tokenization import DeTokenizer

import os

def main(model_cfg='config/bert_base.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         max_len=512,
         max_pred=20,
         mask_prob=0.15,
         GPUs='0'):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    model_cfg = models.Config.from_json(model_cfg)

    bs = 4

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   bs,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    model = BertAEModel4Pretrain(model_cfg)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()
    
    cnt = 0

    for batch in data_iter:
        batch = [t.to(device) for t in batch]
        with torch.no_grad(): # evaluation without gradient calculation
            input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

            logits_lm, logits_clsf, _, _, _ = model(input_ids, segment_ids, input_mask, masked_pos)

            pred_token_ids = torch.argmax(logits_lm, dim=2)
            pred_token_ids = np.array(pred_token_ids.cpu()[0])
            masked_ids     = np.array(masked_ids.cpu()[0])
            correct_pos    = set(np.where(pred_token_ids == masked_ids)[0])
            pred_tokens    = DeTokenizer.convert_ids_to_tokens(pred_token_ids, DeTokenizer.read_dic_to_set(vocab))
            gt_tokens      = DeTokenizer.convert_ids_to_tokens(masked_ids, DeTokenizer.read_dic_to_set(vocab))

            print('Masked Words Prediction   : ', end='')
            for i, token in enumerate(pred_tokens):
                if i in correct_pos:
                    print(f'{Fore.GREEN}{token}{Style.RESET_ALL} ', end='')
                else:
                    print(f'{Fore.RED}{token}{Style.RESET_ALL} ', end='')

            print('')
            
            print('Masked Words Ground-truth : ', end='')
            for i, token in enumerate(gt_tokens):
                if i in correct_pos:
                    print(f'{Fore.GREEN}{token}{Style.RESET_ALL} ', end='')
                else:
                    print(token+' ', end='')

            print('\n')
            
            is_next_pred = bool(torch.argmax(logits_clsf, dim=1)[0])
            is_next_gt   = bool(is_next[0])

            if is_next_pred == is_next_gt:
                print(f'Is_next Prediction   : {Fore.GREEN}{is_next_pred}{Style.RESET_ALL}') 
                print(f'Is_next Ground-truth : {Fore.GREEN}{is_next_gt}{Style.RESET_ALL}') 
            else:
                print(f'Is_next Prediction   : {Fore.RED}{is_next_pred}{Style.RESET_ALL}') 
                print(f'Is_next Ground-truth : {is_next_gt}') 
            
            line = '*-'
            print(f'{Fore.BLUE}{line}{Style.RESET_ALL}'*50)
                
            cnt += 1
            if cnt >= 10:
                break

if __name__ == '__main__':
    fire.Fire(main)
        