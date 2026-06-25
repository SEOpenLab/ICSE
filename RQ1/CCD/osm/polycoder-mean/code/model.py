# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        #x = features[:, -1, :]  # take </s> token (equiv. to [CLS])
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, input_ids=None,attention_mask=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        attention_mask=attention_mask.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids= input_ids,attention_mask=attention_mask)[0]
        #mean-pooling
        attention_mask_mean = attention_mask.clone()
        print("------------------------------attention-mask------------------------")
        print(attention_mask_mean)
        attention_mask_mean[:, -1] = 0
        print("------------------------------attention-mask-mean-----------------------")
        print(attention_mask_mean)
        # mean pooling
        input_mask_expanded = attention_mask_mean.unsqueeze(-1).float()
        output = torch.sum(outputs * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)


        logits = self.classifier(output)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
      
        
 
        


