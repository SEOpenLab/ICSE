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
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args

        self.classifier=RobertaClassificationHead(config)
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(49152))[0]
        output = outputs[:, 0, :]  #隐藏层的第一个标记对应的向量  

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
      
        
 
