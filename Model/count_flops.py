import torch
from ptflops import get_model_complexity_info

import models
from pretrainDexBERT import BertAEModel4Pretrain
from task_modules import PredictionModel

device = 'cuda:3'

def input_constructor(input_shape):
    input_ids = torch.rand(1, 512).long().to(device)
    segment_ids = torch.rand(1, 512).long().to(device)
    input_mask = torch.rand(1, 512).long().to(device)
    masked_pos = torch.rand(1, 20).long().to(device)
    
    return {'input_ids':input_ids, 'segment_ids':segment_ids, 'input_mask':input_mask, 'masked_pos':masked_pos}

if __name__ == "__main__":

    # DexBERT
    model_cfg = models.Config.from_json("config/DexBERT/bert_base.json")
    input_shape = ((1, 512), (1, 512), (1, 512), (1, 20))
    net = BertAEModel4Pretrain(model_cfg).to(device)
    macs, params = get_model_complexity_info(net, input_shape, as_strings=True,
                                            print_per_layer_stat=True, input_constructor=input_constructor, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    # Prediction Model in Downstream Tasks
    model_cfg = models.Config.from_json("config/AE/bert_base.json")
    input_shape = ((1, 128))
    net = PredictionModel().to(device)
    macs, params = get_model_complexity_info(net, input_shape, as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))