from PIL import Image
from torch.utils.data import Dataset
import os
from glob import glob
import pandas as pd
import sklearn.utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np




def get_split(text1):
    '''Get split of the text with 200 char lenght'''
    l_total = []
    l_parcial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return str(l_total)


def preprocess_data(df, tokenizer, SEQ_LEN, mri_report_section, pa_report_section):

    df = df[df[pa_report_section].notna()]
    
    df1 = df[[mri_report_section, pa_report_section]]

    # remove special characters from text column
    df1[mri_report_section] = df1[mri_report_section].str.replace('[#,@,&]', '')
    df1[pa_report_section] = df1[pa_report_section].str.replace('[#,@,&]', '')
  
    df1[mri_report_section] = df1[mri_report_section].str.replace('\s+', ' ')
    df1[pa_report_section] = df1[pa_report_section].str.replace('\s+', ' ')

    pa_data = df1[pa_report_section]

    mri_data = df1[mri_report_section]

    study_ids = df['study_id'].to_list()

    pa_encoding = tokenizer.batch_encode_plus(
    list(pa_data),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )

    mri_encoding = tokenizer.batch_encode_plus(
    list(mri_data),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )

    study_id_to_index = {id_: idx for idx, id_ in enumerate(study_ids)}
    index_to_study_id = {idx: id_ for id_, idx in study_id_to_index.items()}
    study_ids_index = [study_id_to_index[id_] for id_ in study_ids]

    return pa_encoding, mri_encoding, study_ids_index, study_id_to_index, index_to_study_id


def loadData(prep_df, batch_size, num_workers, sampler):
    
    return  DataLoader(
            prep_df,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )

def get_data_loaders(tokenizer, config, csv_path):

    num_workers = config['num_workers'] 
    batch_size = config['batch_size']
    seq_len = config['SEQ_LEN']

    mri_report_section = config['mri_report_section']
    pa_report_section = config['pa_report_section']

    df_predict = pd.read_csv(csv_path)


    pa_testencoding,  mri_testencoding, study_ids_index, study_id_to_index, index_to_study_id = preprocess_data(df_predict, tokenizer, seq_len, mri_report_section, pa_report_section)

    pa_test_seq = torch.tensor(pa_testencoding['input_ids'])
    pa_test_mask = torch.tensor(pa_testencoding['attention_mask'])
    pa_test_token_ids = torch.tensor(pa_testencoding['token_type_ids'])
    mri_test_seq = torch.tensor(mri_testencoding['input_ids'])
    mri_test_mask = torch.tensor(mri_testencoding['attention_mask'])
    mri_test_token_ids = torch.tensor(mri_testencoding['token_type_ids'])
   

    study_ids_tensor = torch.tensor(study_ids_index)

    # wrap tensors
    test_data = TensorDataset(pa_test_seq, pa_test_mask, pa_test_token_ids, mri_test_seq, mri_test_mask, mri_test_token_ids, study_ids_tensor)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # Val Data Loader
    testdata = loadData(test_data, batch_size, num_workers, test_sampler)

    print('Number of data in the test set', len(testdata))
    

    return testdata, df_predict, study_id_to_index, index_to_study_id

