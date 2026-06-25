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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self,  starencoder, codegpt, config,tokenizer,args):
        super(Model, self).__init__()
        self.starencoder = starencoder
        self.codegpt = codegpt

        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        # Adapter（核心）
        self.adapter = nn.Sequential(nn.Linear(768, 1024),nn.GELU(), nn.Linear(1024, 768),nn.GELU(),nn.Linear(768, 768),nn.LayerNorm(768))

        
    def forward(self, input_ids=None,attention_mask=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        attention_mask=attention_mask.view(-1,self.args.block_size)

        starencoder_outputs = self.starencoder(input_ids= input_ids,attention_mask=attention_mask)[0]
        # ===== Adapter（对齐）=====
        adapted = self.adapter(starencoder_outputs) + starencoder_outputs
        # ===== codegpt Encoder=====
        code_outputs = self.codegpt(inputs_embeds=adapted,attention_mask=attention_mask)[0]

        output = code_outputs[:, 0, :]  #隐藏层的最后一个标记对应的向量  
        output = output.reshape(-1,output.size(-1)*2)


        logits=self.classifier(output )
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
      
        
 
        


