# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import pandas as pd
    

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        #x = torch.tanh(x)
        x = self.out_proj(x)
        return x

    
class Model(nn.Module):   
    def __init__(self, bert, polycoder, config,tokenizer,args):
        super(Model, self).__init__()
        self.bert = bert
        self.polycoder = polycoder

        self.config=config
        self.tokenizer=tokenizer
        self.args=args

        hidden_size = config.hidden_size

        # Adapter（核心）
        self.adapter = nn.Sequential(nn.Linear(768, 1024),nn.GELU(), nn.Linear(1024, 768),nn.GELU(),nn.Linear(768, 768),nn.LayerNorm(768))

        self.classifier=RobertaClassificationHead(config)
        
    def forward(self, input_ids=None,attention_mask=None,labels=None): 
        # ===== BERT =====
        bert_outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)[0]   # (B, L, 768)

        # ===== Adapter（对齐）=====
        adapted = self.adapter(bert_outputs) + bert_outputs 

        # ===== CodeBERT Encoder=====
        code_outputs = self.polycoder(inputs_embeds=adapted,attention_mask=attention_mask)[0]

        output = code_outputs[:, 0, :]  #隐藏层的最后一个标记对应的向量  

        logits = self.classifier(output)
        #prob=F.softmax(logits)
        prob=torch.sigmoid(logits)
        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits, labels)
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()

            return loss,prob
        else:
            return prob
      
        
 
