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


def preprocess_data(df, tokenizer, SEQ_LEN, mri_report_section, label):

  
    df1 = df[[mri_report_section, label]]
  
    if label == 'any_cancer_1_3':
        df1[label] = df1[label].astype(np.float64)
    if label == 'cancer_status_a_e':
        label_mapping = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5
        }
        df1[label] = df1[label].replace(label_mapping)
        df1[label] = df1[label].astype(np.float64)
   

    # remove special characters from text column
    df1[mri_report_section] = df1[mri_report_section].str.replace('[#,@,&]', '')
   
  
    df1[mri_report_section] = df1[mri_report_section].str.replace('\s+', ' ')


    mri_data = df1[mri_report_section]

    label = df1[label]
    
    mri_encoding = tokenizer.batch_encode_plus(
    list(mri_data),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True
    )


    return mri_encoding, label


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

    # mri_report_section = config['mri_report_section']

    pa_report_section = config['pa_report_section']

    
    if config['label'] == 'cancer_presence':
        label = 'any_cancer_1_3'
    elif config['label'] == 'cancer_status':
        label = 'cancer_status_a_e'

    df_train = read_data(config['csv_train_path'])
    df_val = read_data(config['csv_val_path'])

    df_combined = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    df_test = read_data(config['csv_test_path'])

    trencoding, label_train = preprocess_data(df_combined, tokenizer, seq_len, pa_report_section, label)
     
    valencoding, label_val = preprocess_data(df_val, tokenizer, seq_len, pa_report_section, label)
    
    testencoding, label_test = preprocess_data(df_test, tokenizer, seq_len, pa_report_section, label)


    
    train_seq = torch.tensor(trencoding['input_ids'])
    train_mask = torch.tensor(trencoding['attention_mask'])
    train_token_ids = torch.tensor(trencoding['token_type_ids'])
    train_y = torch.tensor(label_train.tolist())

    val_seq = torch.tensor(valencoding['input_ids'])
    val_mask = torch.tensor(valencoding['attention_mask'])
    val_token_ids = torch.tensor(valencoding['token_type_ids'])
    val_y = torch.tensor(label_val.tolist())

    test_seq = torch.tensor(testencoding['input_ids'])
    test_mask = torch.tensor(testencoding['attention_mask'])
    test_token_ids = torch.tensor(testencoding['token_type_ids'])
    test_y = torch.tensor(label_test.tolist())

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_token_ids, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # Train Data Loader
    traindata = loadData(train_data, batch_size, num_workers, train_sampler)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_token_ids, val_y)
    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)
    # Val Data Loader
    valdata = loadData(val_data, batch_size, num_workers, val_sampler)

    # wrap tensors
    test_data = TensorDataset(test_seq, test_mask, test_token_ids, test_y)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # Val Data Loader
    testdata = loadData(test_data, batch_size, num_workers, test_sampler)

    print('Number of data in the train set', len(traindata))
    print('Number of data in the validation set', len(valdata))
    print('Number of data in the test set', len(testdata))

    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(df_train[label].values.tolist()), 
                                    y=df_train[label])

    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    

    return traindata, valdata, testdata, weights

