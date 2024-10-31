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


def read_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, tokenizer, SEQ_LEN, mri_report_section, pa_report_section, label):

    # df_nan = df[df[report_section].notna()].copy()
    df1 = df[[mri_report_section, pa_report_section, label]]
    # df1 = df1.rename(columns={label: 'label'})

    if label == 'any_cancer_1_3':
        df1[label] = df1[label].astype(np.float64)
    if label == 'cancer_status_a_e':
        label_mapping = {
            'a': 1,
            'b': 2,
            'c': 3
            # 'd': 4,
            # 'e': 5
        }
        df1[label] = df1[label].replace(label_mapping)
        df1[label] = df1[label].astype(np.float64)

    df1[label].value_counts()
    # remove special characters from text column
    df1[mri_report_section] = df1[mri_report_section].str.replace('[#,@,&]', '')
    df1[pa_report_section] = df1[pa_report_section].str.replace('[#,@,&]', '')
  
    df1[mri_report_section] = df1[mri_report_section].str.replace('\s+', ' ')
    df1[pa_report_section] = df1[pa_report_section].str.replace('\s+', ' ')
  

    pa_data = df1[pa_report_section]

    mri_data = df1[mri_report_section]

    label = df1[label]
 

    pa_encoding = tokenizer.batch_encode_plus(
    list(pa_data),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True
    )

    mri_encoding = tokenizer.batch_encode_plus(
    list(mri_data),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True
    )


    return pa_encoding, mri_encoding, label


def loadData(prep_df, batch_size, num_workers, sampler):
    
    return  DataLoader(
            prep_df,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )

def get_data_loaders(tokenizer, config):

    num_workers = config['num_workers'] 
    batch_size = config['batch_size']
    seq_len = config['SEQ_LEN']

    mri_report_section = config['mri_report_section']
    pa_report_section = config['pa_report_section']

    
    if config['label'] == 'cancer_presence':
        label = 'any_cancer_1_3'
    elif config['label'] == 'cancer_status':
        label = 'cancer_status_a_e'


    df_test = pd.read_csv(config['csv_test_path'])
  
    
    pa_testencoding,  mri_testencoding, label_test = preprocess_data(df_test, tokenizer, seq_len, mri_report_section, pa_report_section, label)


    pa_test_seq = torch.tensor(pa_testencoding['input_ids'])
    pa_test_mask = torch.tensor(pa_testencoding['attention_mask'])
    pa_test_token_ids = torch.tensor(pa_testencoding['token_type_ids'])
    mri_test_seq = torch.tensor(mri_testencoding['input_ids'])
    mri_test_mask = torch.tensor(mri_testencoding['attention_mask'])
    mri_test_token_ids = torch.tensor(mri_testencoding['token_type_ids'])
    test_y = torch.tensor(label_test.tolist())

    # wrap tensors
    test_data = TensorDataset(pa_test_seq, pa_test_mask, pa_test_token_ids, mri_test_seq, mri_test_mask, mri_test_token_ids, test_y)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # Val Data Loader
    testdata = loadData(test_data, batch_size, num_workers, test_sampler)

    print('Number of data in the test set', len(testdata))


    return testdata

