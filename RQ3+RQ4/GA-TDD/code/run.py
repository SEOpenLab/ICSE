# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve, matthews_corrcoef


os.environ['CUDA_VISIBLE_DEVICES']='1'

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel,BertModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer, AutoConfig, AutoTokenizer, AutoModel)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

############################################
# 1. 模型注册（你可以扩展到6个模型）+ 全局缓存（model + tokenizer + config）
############################################


MODEL_REGISTRY = {
    "bert": {
        "path": "./bert-base-cased",
        "layer_key": "encoder.layer",
        "hidden_size": 768
    },
    "codebert": {
        "path": "./codebert-base",
        "layer_key": "encoder.layer",
        "hidden_size": 768
    },
    "starencoder": {
        "path": "./starencoder",
        "layer_key": "encoder.layer",
        "hidden_size": 768
    },
    "codegpt": {
        "path": "./CodeGPT-small-java",
        "layer_key": "h",
        "hidden_size": 768
    },
    "polycoder": {
        "path": "./PolyCoder-160M",
        "layer_key": "layers",
        "hidden_size": 768
    },
    "codegen": {
        "path": "./codegen-350M-multi",
        "layer_key": "h",
        "hidden_size": 1024
    },    
}

############################################
# 4. BLOCK_POOL（结构配置）
############################################

BLOCK_POOL = [

    {"model_id": "bert", "layers": [0, 1]},
    {"model_id": "bert", "layers": [10, 11]},
    {"model_id": "bert", "layers": [6, 7]},
    {"model_id": "codebert", "layers": [4, 5]},
    {"model_id": "codebert", "layers": [0, 1]},
    {"model_id": "codebert", "layers": [2, 3]},

    {"model_id": "starencoder", "layers": [2, 3]},
    {"model_id": "starencoder", "layers": [7, 8]},
    {"model_id": "starencoder", "layers": [4, 5]},
    
    {"model_id": "codegpt", "layers": [7, 8]},
    {"model_id": "codegpt", "layers": [0, 1]},
    {"model_id": "codegpt", "layers": [3, 4]},
    {"model_id": "polycoder", "layers": [8, 9]},
    {"model_id": "polycoder", "layers": [10, 11]},
    {"model_id": "polycoder", "layers": [2, 3]},
    
    {"model_id": "codegen", "layers": [10, 11]},
    {"model_id": "codegen", "layers": [0, 1]},
    {"model_id": "codegen", "layers": [6, 7]},
]
MODEL_CACHE = {}


############################################
# 2. Connector
############################################


class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        mid = max(in_dim, out_dim) * 2
        print("################ in_dim ###########")
        print(in_dim)
        print("################ mid_dim ###########")
        print(mid)
        print("################ out_dim ###########")        
        print(out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Linear(mid, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)        


class GateConnector(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck=256):
        super().__init__()
        
        print("################ in_dim ###########")
        print(in_dim)
        print("################ bottleneck ###########")
        print(bottleneck)
        print("################ out_dim ###########")        
        print(out_dim)
        
        self.proj_in = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.down = nn.Linear(out_dim, bottleneck)
        self.up = nn.Linear(bottleneck, out_dim)

        self.gate = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        residual = x

        h = self.down(x)
        h = F.gelu(h)
        h = self.up(h)

        g = torch.sigmoid(self.gate(x))

        out = g * h + (1 - g) * residual
        return self.norm(out)
        
def build_connector(t, in_dim, out_dim):

    if t == "linear":
        return LinearConnector(in_dim, out_dim)

    elif t == "gate":
        return GateConnector(in_dim, out_dim)

    else:
        raise ValueError("Unknown connector type")

############################################
# 3. 分类头（TDD）会变化 因为codegen是1024,暂时独立出去
############################################

class RobertaClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        #x = self.dense(x)
        x = torch.tanh(x)
        #x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class BlockModel(nn.Module):

    def __init__(self,
                 blocks,
                 connectors,
                 hidden_sizes,
                 first_model_id):

        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.connectors = nn.ModuleList(connectors)
        final_hidden = hidden_sizes[-1]
        self.classifier = RobertaClassificationHead(final_hidden)
        ################################
        # tokenizer/config
        ################################

        self.tokenizer = MODEL_CACHE[first_model_id]["tokenizer"]

        self.config = MODEL_CACHE[first_model_id]["config"]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None):

        ################################
        # 第一个block
        ################################

        x = self.blocks[0](
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

        ################################
        # 后续block
        ################################

            
        for i in range(1, len(self.blocks)):
            x_new = self.connectors[i - 1](x)
            # 只有维度一致才 residual
            if x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new
            x = self.blocks[i](inputs_embeds=x,attention_mask=attention_mask)[0]    

        ################################
        # 分类
        ################################
        output = x[:, 0, :]
        logits = self.classifier(output)
        ################################
        # TDD Loss
        ################################
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + \
                   torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        return prob

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['func'].split()) 
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    if tokenizer.cls_token == tokenizer.sep_token == None:
       source_tokens =[tokenizer.bos_token]+code_tokens+[tokenizer.eos_token]
       source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
       padding_length = args.block_size - len(source_ids)
       attention_mask = [1]*len(source_ids) + [0]*padding_length
       if tokenizer.pad_token_id == None:  
          source_ids+=[tokenizer.eos_token_id]*padding_length
       else:
          source_ids+=[tokenizer.pad_token_id]*padding_length 
         
    else:    
       source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
       source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
       padding_length = args.block_size - len(source_ids)
       attention_mask = [1]*len(source_ids) + [0]*padding_length 
       source_ids+=[tokenizer.pad_token_id]*padding_length

    return InputFeatures(source_tokens,source_ids,attention_mask,js['idx'],js['target']) 

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("attention_mask: {}".format(' '.join(map(str, example.attention_mask))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].attention_mask),torch.tensor(self.examples[i].label)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_GA(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    best_auc=0
    best_f1=0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)

    # Initialize early stopping parameters at the start of training

    early_stopping_counter = 0  # 用于跟踪验证性能未改善的次数
    best_loss = None
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            attention_mask=batch[1].to(args.device)
            labels=batch[2].to(args.device) 
            model.train()
            loss,logits = model(inputs,attention_mask,labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        
                        logger.info(f"early_stopping_counter: {early_stopping_counter}/{args.ga_early_stopping_patience}")#
                        early_stopping_counter = 0  # 重置计数器#
                        logger.info(f" reset early_stopping_counter: {early_stopping_counter}/{args.ga_early_stopping_patience}")#

                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          

                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    else:
                        early_stopping_counter += 1#
                        logger.info(f"early_stopping_counter: {early_stopping_counter}/{args.ga_early_stopping_patience}")#

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num


 
        # 检查是否需要停止训练
        if args.ga_early_stopping_patience is not None:
            if early_stopping_counter >= args.ga_early_stopping_patience:
                logger.info("Early stopping triggered. Stopping training.")
                break  

    return  best_f1 ###############     


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    best_auc=0
    best_f1=0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)

    # Initialize early stopping parameters at the start of training

    early_stopping_counter = 0  # 用于跟踪验证性能未改善的次数
    best_loss = None
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            attention_mask=batch[1].to(args.device)
            labels=batch[2].to(args.device) 
            model.train()
            loss,logits = model(inputs,attention_mask,labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info(f"early_stopping_counter: {early_stopping_counter}/{args.early_stopping_patience}")#
                        early_stopping_counter = 0  # 重置计数器#
                        logger.info(f" reset early_stopping_counter: {early_stopping_counter}/{args.early_stopping_patience}")#

                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          

                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    else:
                        early_stopping_counter += 1#
                        logger.info(f"early_stopping_counter: {early_stopping_counter}/{args.early_stopping_patience}")#

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num


 
        # 检查是否需要停止训练
        if args.early_stopping_patience is not None:
            if early_stopping_counter >= args.early_stopping_patience:
                logger.info("Early stopping triggered. Stopping training.")
                break  

      
        '''
        # Check for early stopping condition
        if args.early_stopping_patience is not None:

            logger.info(f"avg_loss: {avg_loss}")
            logger.info(f"best_loss: {best_loss}")
            logger.info(f"early_stopping_counter: {early_stopping_counter}/{args.early_stopping_patience}")

            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
                logger.info(f"update best_loss: {best_loss}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the loop early
                  
        '''     


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        attention_mask=batch[1].to(args.device)
        label=batch[2].to(args.device)
        with torch.no_grad():
            lm_loss,logit = model(inputs,attention_mask,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    f1=f1_score(labels, preds)
    auc_roc = roc_auc_score(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
        "eval_auc": float(auc_roc),
        "eval_f1": float(f1),
    }
    return result

def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)    
        attention_mask=batch[1].to(args.device)
        label=batch[2].to(args.device) 
        with torch.no_grad():
            logit = model(inputs,attention_mask)###########
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,preds):
            if pred:
                f.write(example.idx+'\t1\n')
            else:
                f.write(example.idx+'\t0\n')    
    
                        
def prepare_tokenizer(tokenizer_class, tokenizer_name, do_lower_case):
    MASK_TOKEN = "<mask>"
    SEPARATOR_TOKEN = "<sep>"
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    try:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    except OSError:
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=do_lower_case, use_auth_token=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer       


############################################
# 7. GA
############################################

CONNECTOR_TYPES = ["linear", "gate"]

MIN_BLOCKS = 2

MAX_BLOCKS = 3


class Chromosome:
    def __init__(self, blocks, connectors):
        self.blocks = blocks
        self.connectors = connectors
        self.fitness = None

    def describe(self):
        desc = []

        for i, b_idx in enumerate(self.blocks):
            cfg = BLOCK_POOL[b_idx]
            model_id = cfg["model_id"]
            layers = cfg["layers"]

            desc.append(f"[{model_id}: layer {layers}]")

            if i < len(self.connectors):
                desc.append(f"--({self.connectors[i]})-->")

        return " ".join(desc)

    def get_tokenizer(self):
        first_block = BLOCK_POOL[self.blocks[0]]
        return first_block["model_id"]
        
    # ⭐ 新增：控制 print() 输出
    ########################################
    def __str__(self):
        return f"Structure: {self.describe()}\nTokenizer: {self.get_tokenizer()}"

    __repr__ = __str__    
############################################
# 初始化
############################################

def random_chromosome():

    n = random.randint(MIN_BLOCKS, MAX_BLOCKS)

    blocks = random.sample(
        range(len(BLOCK_POOL)),
        n
    )

    connectors = [
        random.choice(CONNECTOR_TYPES)
        for _ in range(n - 1)
    ]

    return Chromosome(blocks, connectors)


def init_population(size):

    return [
        random_chromosome()
        for _ in range(size)
    ]
############################################
# 通用取层函数
############################################
def get_layers(model, layer_key):
    keys = layer_key.split(".")
    obj = model
    for k in keys:
        obj = getattr(obj, k)
    return obj
    
############################################
# 5. 构建 Block
############################################

def build_block(cfg):
    model_id = cfg["model_id"]
    layer_ids = cfg["layers"]

    base = MODEL_CACHE[model_id]["model"]
    layer_key = MODEL_REGISTRY[model_id]["layer_key"]

    layers = get_layers(base, layer_key)

    selected_layers = torch.nn.ModuleList(
        [layers[i] for i in layer_ids]
    )

    model = deepcopy(base)

    # ⭐ 把层写回去
    parent = model
    keys = layer_key.split(".")
    for k in keys[:-1]:
        parent = getattr(parent, k)

    setattr(parent, keys[-1], selected_layers)

    model.config.num_hidden_layers = len(selected_layers)

    return model



############################################
# merge consecutive same-model blocks
############################################

def merge_same_model_blocks(cfgs, connectors):

    if len(cfgs) == 0:
        return cfgs, connectors

    merged_cfgs = []

    # connector 是 block 之间的
    # merge 后 connector 数量会变化
    merged_connectors = []

    current = deepcopy(cfgs[0])

    for i in range(1, len(cfgs)):

        prev_model = current["model_id"]
        curr_model = cfgs[i]["model_id"]

        ################################
        # same model -> merge layers
        ################################
        if prev_model == curr_model:

            current["layers"] = (
                current["layers"] +
                cfgs[i]["layers"]
            )

        ################################
        # different model
        ################################
        else:

            merged_cfgs.append(current)

            merged_connectors.append(
                connectors[i - 1]
            )

            current = deepcopy(cfgs[i])

    merged_cfgs.append(current)

    return merged_cfgs, merged_connectors



############################################
# 构建模型
############################################

def build_model(chromosome):
    
    print("\n[Build Model]")
    print("Structure:", chromosome.describe())
    print("Tokenizer:", chromosome.get_tokenizer())
    
    ################################
    # 解析 block 配置
    ################################
    cfgs = [
        deepcopy(BLOCK_POOL[i])
        for i in chromosome.blocks
    ]
    
    ################################
    # merge consecutive same-model blocks
    ################################
    cfgs, merged_connectors = merge_same_model_blocks(
        cfgs,
        chromosome.connectors
    )

    ################################
    # 构建 blocks + 记录 hidden size
    ################################
    blocks = []
    hidden_sizes = []

    shared_embedding = None
    shared_wte = None
    
    for idx, c in enumerate(cfgs):
    
        model_id = c["model_id"]
    
        ################################
        # 第一个 block
        ################################
        if idx == 0:
    
            block = build_block(c)
    
            ################################
            # 保存 embedding 引用
            ################################
    
            if hasattr(block, "embeddings"):
                shared_embedding = block.embeddings
    
            if hasattr(block, "wte"):
                shared_wte = block.wte
    
        ################################
        # 后续 block
        ################################
        else:
    
            block = build_block(c)
    
            ################################
            # 共享 embedding
            ################################
    
            if hasattr(block, "embeddings")and \
               shared_embedding is not None:
    
                print("delete ONLY word_embeddings")
    
                del block.embeddings.word_embeddings
    
                block.embeddings.word_embeddings =  shared_embedding.word_embeddings
                    
            if hasattr(block, "embeddings") and \
               shared_wte is not None:
    
                print("Share shared_wte")
    
                del block.embeddings.word_embeddings
    
                block.embeddings.word_embeddings =  shared_wte               
    
            if hasattr(block, "wte") and \
               shared_embedding is not None:
    
                print("Share embeddings")
    
                del block.wte
    
                block.wte = shared_embedding.word_embeddings
                
            if hasattr(block, "wte") and \
               shared_wte is not None:
    
                print("Share shared_wte")
    
                del block.wte
    
                block.wte = shared_wte                
             
    
        blocks.append(block)
    
        hidden_size = MODEL_REGISTRY[
            model_id
        ]["hidden_size"]
    
        hidden_sizes.append(hidden_size)

    ################################
    # 构建 connectors（关键：自动适配维度）
    ################################
    connectors = []

    for i in range(len(blocks) - 1):
        in_dim = hidden_sizes[i]
        out_dim = hidden_sizes[i + 1]

        connector_type = merged_connectors[i]

        conn = build_connector(
            connector_type,
            in_dim,
            out_dim
        )

        connectors.append(conn)

    ################################
    # 第一个block决定 tokenizer/config
    ################################
    first_model_id = cfgs[0]["model_id"]

    model = BlockModel(
        blocks,
        connectors,
        hidden_sizes,
        first_model_id
    )

    return model
    

############################################
# GA操作
############################################

def select(pop, k=3):
    s = random.sample(pop, k)
    s.sort(
        key=lambda x: x.fitness,
        reverse=True
    )
    return s[0]

############################################
# 交叉
############################################

def crossover(p1, p2):

    cut = random.randint(
        1,
        min(len(p1.blocks),
            len(p2.blocks)) - 1
    )

    blocks = p1.blocks[:cut] + p2.blocks[cut:]

    conns = p1.connectors[:cut - 1] + \
            p2.connectors[cut - 1:]

    return Chromosome(blocks, conns)

############################################
# 变异
############################################

def mutate(c, prob=0.3):

    ################################
    # block mutation
    ################################

    if random.random() < prob:

        i = random.randint(
            0,
            len(c.blocks) - 1
        )

        c.blocks[i] = random.randint(
            0,
            len(BLOCK_POOL) - 1
        )

    ################################
    # connector mutation
    ################################

    if random.random() < prob and \
       len(c.connectors) > 0:

        i = random.randint(
            0,
            len(c.connectors) - 1
        )

        c.connectors[i] = random.choice(
            CONNECTOR_TYPES
        )

    return c
    
    

    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
                             
    parser.add_argument("--do_GA", action='store_true',
                        help="Whether to run GA.")                        
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
                        
                        
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--ga_early_stopping_patience", type=int, default=1,
                        help="Number of epochs with no improvement after which training will be stopped.")                        
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
  
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))


    
        
    for model_id, info in MODEL_REGISTRY.items():
        path = info["path"]
        print(f"Loading {model_id} from {path}")
        
        config = AutoConfig.from_pretrained(path)
        
        if model_id == "starencoder":
            tokenizer = prepare_tokenizer(
                tokenizer_class=AutoTokenizer,
                tokenizer_name=path,
                do_lower_case=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                do_lower_case=True
            )
        
        model = AutoModel.from_pretrained(path, config=config)
        
        MODEL_CACHE[model_id] = {
            "model": model,
            "config": config,
            "tokenizer": tokenizer
        }
    
    

    ################################
    # 初始化
    ################################
    pop_size=3
    generations=3
    top_k=1
    all_candidates = []
    pop = init_population(pop_size)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)




    # Training
    if args.do_GA:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache        
        if args.local_rank == 0:
            torch.distributed.barrier()
        
        ################################
        # GA迭代
        ################################
        for gen in range(generations):
            print(f"\n=== Generation {gen} ===")
            print("################## All Population ##################")
            for i, c in enumerate(pop):
                print(f"[{i}] Fitness: {c.fitness}")
                print(c)
            ################################
            # 评估
            ################################
            for c in pop:
                if c.fitness is None:
                   model = build_model(c)
                   train_dataset = TextDataset(model.tokenizer, args,args.train_data_file)  
                   c.fitness = train_GA(args, train_dataset, model, model.tokenizer) # 适应度函数 - 轻量训练
                all_candidates.append(deepcopy(c))
            pop.sort(key=lambda x: x.fitness, reverse=True) # 排序
            print("################## Current Best ##################")
            print("################## Best Fitness:", pop[0].fitness)
            print("################## Blocks:", pop[0].blocks)
            print("################## Connectors:", pop[0].connectors)
            print("################## Structure:", pop[0])
            
            new_pop = pop[:2] # elitism
            # 生成新种群
            print("################## Generate New Population and Crossover Mutate ################" )
            while len(new_pop) < pop_size:
                p1 = select(pop)
                p2 = select(pop)
                print("################## Crossover and Mutate ################" )
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
            pop = new_pop
        
        print("################## Final Best Top-k ##################")        
        print("################## Final Best Top-k ##################")


            
    all_candidates.sort(key=lambda x: x.fitness, reverse=True)
    top_k_chromosomes = all_candidates[:top_k]    
    models = []
    tokenizers = []    
    for i, c in enumerate(top_k_chromosomes):
        print(f"[Top {i}] Fitness: {c.fitness}")
        print(c)
        model = build_model(c)
        tokenizer = model.tokenizer
        
        models.append(model)
        tokenizers.append(tokenizer)


    model = models[0]
    tokenizer = tokenizers[0]
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache        
        if args.local_rank == 0:
            torch.distributed.barrier()
  
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)        
       
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result=evaluate(args, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test(args, model, tokenizer)

    return results
            


if __name__ == "__main__":
    main()


