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
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = self.dropout(features)
        return self.out_proj(x)


class Model(nn.Module):
    def __init__(self, bert, starencoder, config, tokenizer, args):
        super().__init__()
        self.bert = bert
        self.starencoder = starencoder

        self.args = args
        self.config = config

        hidden = config.hidden_size

        # ✔ 保留你当前最优 adapter
        self.adapter = nn.Sequential(
            nn.Linear(hidden, 1024),
            nn.GELU(),
            nn.Linear(1024, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden)
        )

        self.classifier = RobertaClassificationHead(config)

        # ⭐ alignment projection (optional but useful)
        self.align_proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        # ===== BERT =====
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

        # ===== Adapter =====
        adapted = self.adapter(bert_outputs)

        # ===== StarEncoder =====
        code_outputs = self.starencoder(
            inputs_embeds=adapted,
            attention_mask=attention_mask
        )[0]

        cls = code_outputs[:, 0, :]
        logits = self.classifier(cls)

        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float()

            # ===== main loss =====
            ce_loss = -(
                labels * torch.log(prob[:, 0] + 1e-10) +
                (1 - labels) * torch.log(1 - prob[:, 0] + 1e-10)
            ).mean()

            # ===== alignment loss (关键) =====
            bert_cls = bert_outputs[:, 0, :]
            star_cls = cls

            align_loss = F.mse_loss(
                self.align_proj(bert_cls),
                star_cls.detach()   # 稳定训练
            )

            loss = ce_loss + 0.1 * align_loss

            return loss, prob

        return prob
        
 
