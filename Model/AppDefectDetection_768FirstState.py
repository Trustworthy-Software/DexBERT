import os
from turtle import forward
import fire
import pickle
from jax import debug_infs
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from random import random

import models
import optim
import train

from MaliciousCodeLocalization import Trainer as BaseTrainer
from task_modules import Preprocess4EmbeddingIntegration, ClassSeqDataLoader, Config
from utils import set_seeds, get_device
import tokenization

class BertAEModel(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)

        # auto-encoder
        self.AE_Layer_1 = nn.Linear(cfg.max_len*cfg.dim, cfg.max_len)
        self.AE_Layer_2 = nn.Linear(cfg.max_len, cfg.class_vec_len)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)

        # auto-encoder
        r1 = torch.flatten(h, start_dim=1)
        x = self.AE_Layer_1(r1)
        r2 = self.AE_Layer_2(x)

        return r2, h

class ClassEmbeddingLoader():
    '''Load class embedding generated from pre-trained BERT model.'''
    def __init__(self, malware_samp_file, sample_list, batch_size, postfix, shuffle_list=True, benign_ratio=0.5):
        super().__init__()
        self.apk_list = [x.split(',')[0] for x in open(malware_samp_file, 'r').readlines()]
        # self.class_set = set(['/'.join(x.strip().split('/')[1:]) for x in open(sample_list, 'r').readlines()])
        self.class_set = set(sample_list)
        self.batch_size = batch_size
        self.benign_ratio = benign_ratio
        self.shuffle_list = shuffle_list
        self.postfix = postfix
        
        if shuffle_list:
            self.apk_list = shuffle(self.apk_list)
    
    def read_class_emb(self):
        for apk_bin in self.apk_list:
            version_name = apk_bin.split('/')[-1].split('.txt')[0]
            # apk_emb = pickle.load(open(apk_bin.replace('.txt', '.loc.pkl'), 'rb'))
            apk_emb = pickle.load(open(apk_bin.replace('.txt', self.postfix), 'rb'))
            class_vecs, class_labels, class_names, _ = apk_emb
            sample_list = [(x, y, z) for x, y, z in zip(class_vecs, class_labels, class_names)]
            if self.shuffle_list:
                sample_list = shuffle(sample_list)
            for class_emb, class_label, class_name in sample_list:
                class_name = version_name+'/'+class_name+'.smali'
                if class_name in self.class_set:
                    if class_label == 1:
                        # import ipdb; ipdb.set_trace()
                        yield class_emb, class_label, class_name
                    else:
                        if random() < self.benign_ratio:
                            # import ipdb; ipdb.set_trace()
                            yield class_emb, class_label, class_name
                        else:
                            continue
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


class MaliciousClassDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        prediction = self.classifier(x)
        return prediction


class Trainer(BaseTrainer):
    '''MIL Training Helper'''
    def __init__(self, MIL_train_cfg, BertAEmodel, MalClassModel, optimizer_MIL, save_dir, log_dir, device):
        super(Trainer, self).__init__(MIL_train_cfg, BertAEmodel, None, optimizer_MIL, save_dir, log_dir, device)
        self.cfg = MIL_train_cfg # config for training : see class Config in train.py
        self.BertAEmodel = BertAEmodel
        self.MalClassModel = MalClassModel
        # self.optimizer_BertAE = optimizer_BertAE
        self.optimizer_MIL = optimizer_MIL
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.device = device # device name
        self.apk_list = []

    def compute_class_embeddings(self, data_file, vocab, postfix='.loc.pkl', token_max_len=512, data_parallel=True):
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

                    r2, h = self.BertAEmodel(input_ids, segment_ids, input_mask)
                    # batch_vec = r2.cpu().detach().numpy()
                    batch_vec = h.cpu().detach().numpy()
                    
                    for i, emb in enumerate(batch_vec):
                        if len(class_vector_list) == 0:
                            class_vector_list.append(np.expand_dims(emb[0], axis=0))
                            class_label_list.append(int(class_label.cpu()[i]))
                            class_name_list.append(class_names[i])
                            continue
                        if int(class_id.cpu()[i]) == last_class_id:
                            class_vector_list[-1] = np.concatenate([class_vector_list[-1], np.expand_dims(emb[0], axis=0)])
                            continue
                        class_vector_list.append(np.expand_dims(emb[0], axis=0))
                        class_label_list.append(int(class_label[i].cpu()))
                        class_name_list.append(class_names[i])
                        last_class_id = int(class_id.cpu()[i])

            with open(apk_file.replace('.txt', postfix), 'wb') as f:
                apk_label = max(class_label_list)
                pickle.dump([class_vector_list, class_label_list, class_name_list, apk_label], f)

    def train(self, data_file, train_list, vocab, batch_size, recompute_class_embeddings, token_max_len=512, postfix='.loc768.pkl'):
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
            
            data_loader = ClassEmbeddingLoader(data_file, train_list, batch_size, postfix=postfix)
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

    def evaluate(self, eval_data_file, test_list, model_weights, vocab, recompute_class_embeddings=False, training_mode=False, token_max_len=512, postfix='.loc768.pkl'):
        if recompute_class_embeddings:
            self.compute_class_embeddings(eval_data_file, vocab, postfix, token_max_len)
        
        if not training_mode:
            self.MalClassModel.load_state_dict(torch.load(model_weights), strict=False)
            self.MalClassModel = self.MalClassModel.to(self.device)
        self.MalClassModel.eval()
                
        data_loader = ClassEmbeddingLoader(eval_data_file, test_list, self.cfg.MCD_batch_size, postfix, shuffle_list=False, benign_ratio=1.0)
        apk_iter_bar = tqdm(data_loader, desc='Iter (loss=X.XXX)')

        pre_list = []
        gt_list  = []
        score_list = []
        for i, sample in enumerate(apk_iter_bar): 
            class_info, class_names = sample
            class_vec_batch, class_label_batch = class_info

            class_vec_batch = [torch.sum(class_seqs, 0) for class_seqs in class_vec_batch]
            class_vec_batch = [t.to(self.device) for t in class_vec_batch]
            class_vec_batch = torch.stack((class_vec_batch))

            logits = self.MalClassModel(class_vec_batch)
            pre_batch = torch.argmax(logits, dim=1).detach().cpu().numpy()
            defect_scores = logits.detach().cpu().numpy()[:, 1]

            pre_list.extend(pre_batch.tolist())
            score_list.extend(defect_scores.tolist())
            gt_list.extend(class_label_batch.cpu().numpy().tolist())
        precision, recall, fbeta_score, support = precision_recall_fscore_support(gt_list, pre_list, beta=1.0)

        ROC_AUC_SCORE = roc_auc_score(gt_list, score_list)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        eval_res_file = open(os.path.join(self.log_dir, 'evaluation.txt'), 'w')
        print('  Category\tPre\tRec\tF1\tSamp_num')
        eval_res_file.write('  Category\tPre\tRec\tF1\tSamp_num\n')
        print('Benignware\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        eval_res_file.write('Benignware\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        print('   Malware\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        eval_res_file.write('   Malware\t{:.4f}\t{:.4f}\t{:.4f}\t{}\n'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        print('ROC AUC Score: {:.4f}'.format(ROC_AUC_SCORE))
        eval_res_file.write('ROC AUC Score: {:.4f}\n'.format(ROC_AUC_SCORE))
        eval_res_file.close()
        print(support)
        print(sum(support))
        print(fbeta_score)
        print(ROC_AUC_SCORE)

        return ROC_AUC_SCORE
        

def main(Bert_train_cfg='config/AE_V3/pretrain.json',
         Bert_model_cfg='config/AE_V3/bert_base.json',
         MIL_cfg='config/MCD/MCD.json',
         train_file='../Data/data/defect/data_file_list/AnkiDroid.txt',
         test_file='../Data/data/defect/data_file_list/AnkiDroid.txt',
         train_list='../Data/data/WPDP/AnkiDroid/random_part_0_1_2_3.txt',
         test_list='../Data/data/WPDP/AnkiDroid/random_part_4.txt',
         BertAEmodel_file='../save_dir/AutoEncoderV3/model_steps_604364.pt',
         MalClassModel_file='../save_dir/app_defect_detection/model_steps_347955.pt',
         vocab='../Data/data/pre-train/vocab.txt',
         save_dir='../save_dir/app_defect_detection_5Kep_AEV3',
         log_dir='../log_dir/app_defect_detection_5Kep_AEV3',
         max_len=512,
         GPUs='0',
         recompute_class_embeddings=False,
         training_mode=False,
         testing_mode=False,
         benign_ratio=1.0):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUs)

    Bert_train_cfg = train.Config.from_json(Bert_train_cfg)
    Bert_model_cfg = models.Config.from_json(Bert_model_cfg)
    MIL_cfg  = Config.from_json(MIL_cfg)

    set_seeds(Bert_train_cfg.seed)

    if recompute_class_embeddings:
        BertAE = BertAEModel(Bert_model_cfg)
        BertAE.load_state_dict(torch.load(BertAEmodel_file), strict=False)
    else:
        BertAE = None

    MalClassModel = MaliciousClassDetector()
    optimizer_MIL = optim.optim4GPU(MIL_cfg, MalClassModel)
    
    trainer = Trainer(MIL_cfg, BertAE, MalClassModel, optimizer_MIL, save_dir, log_dir, get_device())
    if training_mode:
        train_list = ['/'.join(x.strip().split('/')[1:]) for x in open(train_list, 'r').readlines()]
        trainer.train(train_file, train_list, vocab, MIL_cfg.MCD_batch_size, recompute_class_embeddings, max_len, float(benign_ratio))
    if testing_mode:
       test_list = ['/'.join(x.strip().split('/')[1:]) for x in open(test_list, 'r').readlines()]
       ROC_AUC_SCORE = trainer.evaluate(test_file, test_list, MalClassModel_file, vocab, recompute_class_embeddings, training_mode, max_len)

if __name__ == '__main__':

    fire.Fire(main)
