import os
from turtle import forward
import fire
import pickle
from jax import debug_infs
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from random import random

import models
import optim
import train
import tokenization

from task_modules import Config, BertAEModel, ClassSeqDataLoader, Preprocess4EmbeddingIntegration, PredictionModel
from utils import set_seeds, get_device


class ClassEmbeddingLoader():
    '''Load class embedding generated from pre-trained BERT model.'''
    def __init__(self, malware_samp_file, batch_size, postfix, shuffle_list=True, benign_ratio=0.01):
        super().__init__()
        self.apk_list = [x.split(',')[0] for x in open(malware_samp_file, 'r').readlines()]
        self.batch_size = batch_size
        self.benign_ratio = benign_ratio
        self.shuffle_list = shuffle_list
        self.postfix = postfix
        
        if shuffle_list:
            self.apk_list = shuffle(self.apk_list)
    
    def read_class_emb(self):
        for apk_bin in self.apk_list:
            apk_emb = pickle.load(open(apk_bin.replace('.txt', self.postfix), 'rb'))
            class_vecs, class_labels, class_names, _ = apk_emb
            sample_list = [(x, y, z) for x, y, z in zip(class_vecs, class_labels, class_names)]
            if self.shuffle_list:
                sample_list = shuffle(sample_list)
            for class_emb, class_label, class_name in sample_list:
                if class_label == 1:
                    yield class_emb, class_label, class_name
                else:
                    if random() < self.benign_ratio:
                        yield class_emb, class_label, class_name
                    else:
                        continue
    
    def __iter__(self):
        def process_batch(batch):
            zip_batch = list(zip(*batch))
            vec_batch = [torch.tensor(x, dtype=torch.float) for x in zip_batch[0]]
            label_batch = torch.tensor(zip_batch[1], dtype=torch.long)
            batch_tensors = [vec_batch, label_batch]
            return batch_tensors
        
        samp_iterator = self.read_class_emb()
        flag = True
        while flag:
            batch = []
            batch_class_names = []
            for _ in range(self.batch_size):
                try:
                    class_emb, class_label, class_name = next(samp_iterator)
                except:
                    flag = False
                    break
                instance = (class_emb, class_label)
                batch.append(instance)
                batch_class_names.append(class_name)
            if len(batch) == 0:
                continue
            # To Tensor
            batch_tensors = process_batch(batch)
            yield batch_tensors, batch_class_names


class Trainer():
    '''MIL Training Helper'''
    def __init__(self, MIL_train_cfg, BertAEmodel, MalClassModel, optimizer_MIL, save_dir, log_dir, device):
        super(Trainer, self).__init__(MIL_train_cfg, BertAEmodel, None, optimizer_MIL, save_dir, log_dir, device)
        self.cfg = MIL_train_cfg # config for training : see class Config in train.py
        self.BertAEmodel = BertAEmodel
        self.MalClassModel = MalClassModel
        self.optimizer_MIL = optimizer_MIL
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device # device name
        self.apk_list = []

    def compute_class_embeddings(self, data_file, vocab, postfix='.loc.pkl', token_max_len=512, data_parallel=True, DataLoader=None):
        self.BertAEmodel.eval() # evaluation mode
        self.BertAEmodel = self.BertAEmodel.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            self.BertAEmodel = nn.DataParallel(self.BertAEmodel)
        
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
        tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))  

        pipeline = [Preprocess4EmbeddingIntegration(tokenizer.convert_tokens_to_ids)]


        self.apk_list = apk_list = open(data_file, 'r').readlines()
        print('compute and save class seq embeddings ...')
        apk_iter_bar = tqdm(apk_list)
        for _, sample in enumerate(apk_iter_bar):
            apk_file, label_file = sample.strip().split(',')
            malicious_classes = [x.strip() for x in open(label_file, 'r').readlines()]
            dataloader = ClassSeqDataLoader(apk_file, malicious_classes, self.cfg.Bert_batch_size, tokenize, token_max_len, pipeline)
        
            class_vector_list = []
            class_label_list = []
            class_name_list = []
            last_class_id  = 0

            seq_iter_bar  = tqdm(dataloader)
            with torch.no_grad():
                for _, meta in enumerate(seq_iter_bar):
                    batch, class_names = meta
                    batch = [t.to(self.device) for t in batch]
                    input_ids, segment_ids, input_mask, class_id, class_label = batch

                    r2 = self.BertAEmodel(input_ids, segment_ids, input_mask)
                    batch_vec = r2.cpu().detach().numpy()
                    
                    for i, emb in enumerate(batch_vec):
                        if len(class_vector_list) == 0:
                            class_vector_list.append(np.expand_dims(emb, axis=0))
                            class_label_list.append(int(class_label.cpu()[i]))
                            class_name_list.append(class_names[i])
                            continue
                        if int(class_id.cpu()[i]) == last_class_id:
                            class_vector_list[-1] = np.concatenate([class_vector_list[-1], np.expand_dims(emb, axis=0)])
                            continue
                        class_vector_list.append(np.expand_dims(emb, axis=0))
                        class_label_list.append(int(class_label[i].cpu()))
                        class_name_list.append(class_names[i])
                        last_class_id = int(class_id.cpu()[i])

            with open(apk_file.replace('.txt', postfix), 'wb') as f:
                # apk_label = max(class_label_list)
                apk_label = 0
                pickle.dump([class_vector_list, class_label_list, class_name_list, apk_label], f)

    def train(self, data_file, vocab, batch_size, recompute_class_embeddings, token_max_len=512, postfix='.locAE128.pkl'):
        """ Train Loop """
        if recompute_class_embeddings:
            self.compute_class_embeddings(data_file, vocab, postfix, token_max_len)
            torch.cuda.empty_cache()
                
        self.MalClassModel.train() # train mode
        self.MalClassModel = self.MalClassModel.to(self.device)
        
        criterion = nn.CrossEntropyLoss()

        global_step = 1 # global iteration steps regardless of epochs
        writer = SummaryWriter(log_dir=self.log_dir)
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            
            data_loader = ClassEmbeddingLoader(data_file, batch_size, postfix=postfix)
            apk_iter_bar = tqdm(data_loader, desc='Iter (loss=X.XXX)')

            for i, sample in enumerate(apk_iter_bar): 
                class_info, class_names = sample
                class_vec_batch, class_label_batch = class_info

                class_vec_batch = [torch.sum(class_seqs, 0) for class_seqs in class_vec_batch]
                class_vec_batch = [t.to(self.device) for t in class_vec_batch]
                class_vec_batch = torch.stack((class_vec_batch))
                class_label_batch = [t.to(self.device) for t in class_label_batch]
                class_label_batch = torch.stack((class_label_batch))

                self.optimizer_MIL.zero_grad()
                logits = self.MalClassModel(class_vec_batch)
                loss = criterion(logits, class_label_batch)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer_MIL.step()

                apk_iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                writer.add_scalars('data/scalar_group',
                                {'loss_MalClassPre': loss.item(),
                                'lr': self.optimizer_MIL.get_lr()[0],
                                },
                                global_step)
                
                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

                global_step += 1

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def save(self, i):
        """ save current model """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.MalClassModel.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

    def evaluate(self, eval_data_file, model_weights, vocab, recompute_class_embeddings=False, training_mode=True, token_max_len=512, postfix='.locAE128.pkl'):
        if recompute_class_embeddings:
            self.compute_class_embeddings(eval_data_file, vocab, postfix, token_max_len)
            torch.cuda.empty_cache()
        
        if not training_mode:
            self.MalClassModel.load_state_dict(torch.load(model_weights), strict=False)
            self.MalClassModel = self.MalClassModel.to(self.device)
        self.MalClassModel.eval()
                
        data_loader = ClassEmbeddingLoader(eval_data_file, self.cfg.MCD_batch_size, postfix, shuffle_list=False, benign_ratio=0.05)
        apk_iter_bar = tqdm(data_loader, desc='Iter (loss=X.XXX)')

        pre_list = []
        gt_list  = []
        for i, sample in enumerate(apk_iter_bar): 
            class_info, class_names = sample
            class_vec_batch, class_label_batch = class_info

            class_vec_batch = [torch.sum(class_seqs, 0) for class_seqs in class_vec_batch]
            class_vec_batch = [t.to(self.device) for t in class_vec_batch]
            class_vec_batch = torch.stack((class_vec_batch))

            logits = self.MalClassModel(class_vec_batch)
            pre_batch = torch.argmax(logits, dim=1).detach().cpu().numpy()

            pre_list.extend(pre_batch.tolist())
            gt_list.extend(class_label_batch.cpu().numpy().tolist())
        precision, recall, fbeta_score, support = precision_recall_fscore_support(gt_list, pre_list, beta=1.0)

        FP = len(np.where((np.array(pre_list) == 1) & (np.array(gt_list) == 0))[0])
        TN = len(np.where((np.array(pre_list) == 0) & (np.array(gt_list) == 0))[0])
        FPR = FP / (FP + TN)

        FN = len(np.where((np.array(pre_list) == 0) & (np.array(gt_list) == 1))[0])
        TP = len(np.where((np.array(pre_list) == 1) & (np.array(gt_list) == 1))[0])
        FNR = FN / (FN + TP)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        eval_res_file = open(os.path.join(self.log_dir, 'evaluation.txt'), 'w')
        print('  Category\tPre\tRec\tF1\tSamp_num')
        eval_res_file.write('  Category\tPre\tRec\tF1\tSamp_num\n')
        print('Benignware\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        eval_res_file.write('Benignware\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        print('   Malware\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        eval_res_file.write('   Malware\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        print('False Negative Rate: {:.4f}'.format(FNR))
        eval_res_file.write('False Negative Rate: {:.4f}\n'.format(FNR))
        print('False Positive Rate: {:.4f}'.format(FPR))
        eval_res_file.write('False Positive Rate: {:.4f}\n'.format(FPR))
        eval_res_file.close()
        

def main(Bert_train_cfg='config/DexBERT/retrain.json',
         Bert_model_cfg='config/DexBERT/bert_base.json',
         MIL_cfg='config/MCL/MCL.json',
         data_file='../Data/data/localization_train_samples.txt',
         BertAEmodel_file='../save_dir/DexBERT/model_steps_604364.pt',
         AE_model_file='768AE128.pth',
         MalClassModel_file='../save_dir/malicious_code_localization/model_steps_4076.pt',
         vocab='../Data/data/pre-train/vocab.txt',
         save_dir='../save_dir/malicious_code_localization',
         log_dir='../log_dir/malicious_code_localization',
         max_len=512,
         GPUs='0',
         recompute_class_embeddings=True,
         training_mode=True,
         testing_mode=True,
         time_cost=False):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUs)

    Bert_train_cfg = train.Config.from_json(Bert_train_cfg)
    Bert_model_cfg = models.Config.from_json(Bert_model_cfg)
    MIL_cfg  = Config.from_json(MIL_cfg)

    set_seeds(Bert_train_cfg.seed)

    if recompute_class_embeddings:
        BertAE = BertAEModel(Bert_model_cfg)

        # BertAE.load_state_dict(torch.load(BertAEmodel_file), strict=False)

        pretrained_state_dict = torch.load(BertAEmodel_file)
        # Filter out the keys corresponding to the layer you want to skip
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if "AE_Layer_1" not in k and "AE_Layer_2" not in k}
        BertAE.load_state_dict(filtered_state_dict, strict=False)
        device = torch.device("cuda:0")
        BertAE.load_state_dict(torch.load(AE_model_file, map_location=device), strict=False)
    else:
        BertAE = None

    MalClassModel = PredictionModel()
    optimizer_MIL = optim.optim4GPU(MIL_cfg, MalClassModel)
    
    trainer = Trainer(MIL_cfg, BertAE, MalClassModel, optimizer_MIL, save_dir, log_dir, get_device())
    if training_mode:
        trainer.train(data_file, vocab, MIL_cfg.MCD_batch_size, recompute_class_embeddings, max_len, postfix='.locAE128.pkl')
    if testing_mode:
        trainer.evaluate(data_file.replace('train', 'test'), MalClassModel_file, vocab, recompute_class_embeddings, training_mode, max_len, postfix='.locAE128.pkl')
    if time_cost:
        trainer.compute_infer_time(data_file, vocab, token_max_len=max_len)

if __name__ == '__main__':

    fire.Fire(main)
