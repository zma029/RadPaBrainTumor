import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import os
from transformers import BertModel, get_linear_schedule_with_warmup, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ClinicalBertModel(nn.Module):
    
    def __init__(self, n_classes, model_name, freeze_bert=True):
        
        super(ClinicalBertModel, self).__init__()
        # Instantiating BERT model object
        # self.bert = BertModel.from_pretrained(model_name, return_dict=False)

        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
                
        self.bert_drop_1 = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * self.bert.config.hidden_size, self.bert.config.hidden_size) # (768, 64)
        self.bn = nn.BatchNorm1d(self.bert.config.hidden_size) # (768)
        self.bert_drop_2 = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) # (768,3)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)


    def forward(self, input_ids, attention_mask, token_type_ids, input_ids2, attention_mask2, token_type_ids2):
        output1 = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        output2 = self.bert(
            input_ids = input_ids2,
            attention_mask = attention_mask2,
            token_type_ids = token_type_ids2
        )

        sentence_embeddings1 = mean_pooling(output1, attention_mask)

        sentence_embeddings2 = mean_pooling(output2, attention_mask2)

        sentence_embeddings = torch.cat((sentence_embeddings1, sentence_embeddings2), dim=1)
        
        output = self.bert_drop_1(sentence_embeddings)
        output = self.fc(output)
        output = self.bn(output)
        output = self.bert_drop_2(output)
        output = self.out(output)        
        return output


    


    