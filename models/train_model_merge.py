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
                # print(p.requires_grad)
                p.requires_grad = False

        # self.attention_pathology = nn.Linear(self.bert.config.hidden_size, 1)
        # self.attention_mri = nn.Linear(self.bert.config.hidden_size, 1)
                
        self.bert_drop_1 = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * self.bert.config.hidden_size, self.bert.config.hidden_size) 
        self.bn = nn.BatchNorm1d(self.bert.config.hidden_size) 
        self.bert_drop_2 = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)


    def forward(self, pa_ids, pa_mask, pa_token_type_ids, mri_ids, mri_mask, mri_token_type_ids):

        outputs_patho = self.bert(
            input_ids = pa_ids,
            attention_mask = pa_mask,
            token_type_ids = pa_token_type_ids
        )
        hidden_state_pathology = outputs_patho.last_hidden_state
        pooler_output_pathology = outputs_patho.pooler_output

        outputs_mri = self.bert(
            input_ids = mri_ids,
            attention_mask = mri_mask,
            token_type_ids = mri_token_type_ids
        )
        hidden_state_mri = outputs_mri.last_hidden_state
        pooler_output_mri = outputs_mri.pooler_output

        

        # cross attention
        # attention pathology to mri
        # attention_scores_P2M = torch.matmul(hidden_state_pathology, hidden_state_mri.transpose(-1, -2))
        # attention_weights_P2M = F.softmax(attention_scores_P2M, dim=-1)
        # # Weighted sum of B's encodings to get the attended context of A
        # attended_MRI = torch.matmul(attention_weights_P2M, hidden_state_mri)

        # attention_scores_M2P = torch.matmul(hidden_state_mri, hidden_state_pathology.transpose(-1, -2))
        # attention_weights_M2P = F.softmax(attention_scores_M2P, dim=-1)
        # # Weighted sum of B's encodings to get the attended context of A
        # attended_Pat = torch.matmul(attention_weights_M2P, hidden_state_mri)

        # combined_features_P = torch.cat((attended_Pat.mean(1), pooler_output_pathology), dim=1)
        # combined_features_M = torch.cat((attended_MRI.mean(1), pooler_output_mri), dim=1)

        # combined_features = torch.cat((attended_MRI.mean(1), attended_Pat.mean(1)), dim=1)
        
        # self attention
        # attention_weights_pathology = F.softmax(self.attention_pathology(hidden_state_pathology), dim=1)
        # attention_weights_mri = F.softmax(self.attention_mri(hidden_state_mri), dim=1)

        # weighted_average_pathology = torch.sum(attention_weights_pathology * hidden_state_pathology, dim=1)
        # weighted_average_mri = torch.sum(attention_weights_mri * hidden_state_mri, dim=1)

        # combined_attention_features = torch.cat((weighted_average_pathology, weighted_average_mri), dim=1)


        ###################### sentence embedding
        sentence_embeddings1 = mean_pooling(outputs_patho, pa_mask)

        sentence_embeddings2 = mean_pooling(outputs_mri, mri_mask)
        sentence_embeddings = torch.cat((sentence_embeddings1, sentence_embeddings2), dim=1)
        
        output = self.bert_drop_1(sentence_embeddings)
        output = self.fc(output)
        output = self.bn(output)
        output = self.bert_drop_2(output)
        output = self.out(output)        
        return output


    


    