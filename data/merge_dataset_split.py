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
    df1 = df1.rename(columns={label: 'label'})

    if label == 'any_cancer_1_3':
        df1['label'] = df1['label'].astype(np.float64)
    if label == 'cancer_status_a_e':
        label_mapping = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5
        }
        df1['label'] = df1['label'].replace(label_mapping)
        df1['label'] = df1['label'].astype(np.float64)

    df1 = sklearn.utils.shuffle(df1)

    # remove special characters from text column
    df1[mri_report_section] = df1[mri_report_section].str.replace('[#,@,&]', '')
    df1[pa_report_section] = df1[pa_report_section].str.replace('[#,@,&]', '')
  
    df1[mri_report_section] = df1[mri_report_section].str.replace('\s+', ' ')
    df1[pa_report_section] = df1[pa_report_section].str.replace('\s+', ' ')
  

    indices = df1.index

    # 分割索引
    indices_train, indices_test = train_test_split(indices, test_size=0.1, shuffle=True, random_state=42)

    # 使用分割后的索引从 pa_report_section 和 mri_report_section 列获取数据
    pa_data_train = df1.loc[indices_train, pa_report_section]
    pa_data_test = df1.loc[indices_test, pa_report_section]

    mri_data_train = df1.loc[indices_train, mri_report_section]
    mri_data_test = df1.loc[indices_test, mri_report_section]

    # 同样方法分割标签
    label_train = df1.loc[indices_train, 'label']
    label_test = df1.loc[indices_test, 'label']

    indices_train_final, indices_val = train_test_split(indices_train, test_size=0.2, random_state=42)

    # 使用更新后的训练索引和验证索引来获取对应的数据
    pa_data_train_final = df1.loc[indices_train_final, pa_report_section]
    pa_data_val = df1.loc[indices_val, pa_report_section]

    mri_data_train_final = df1.loc[indices_train_final, mri_report_section]
    mri_data_val = df1.loc[indices_val, mri_report_section]

    label_train_final = df1.loc[indices_train_final, 'label']
    label_val = df1.loc[indices_val, 'label']
    

    pa_trencoding = tokenizer.batch_encode_plus(
    list(pa_data_train),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True
    )

    mri_trencoding = tokenizer.batch_encode_plus(
    list(mri_data_train),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True
    )

    pa_valencoding = tokenizer.batch_encode_plus(
    list(pa_data_val),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )

    mri_valencoding = tokenizer.batch_encode_plus(
    list(mri_data_val),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )


    pa_testencoding = tokenizer.batch_encode_plus(
    list(pa_data_test),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )

    mri_testencoding = tokenizer.batch_encode_plus(
    list(mri_data_test),
    max_length=SEQ_LEN,
    # add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    truncation=True,
    padding='longest',
    return_attention_mask=True,
    )

    return pa_trencoding, pa_valencoding, pa_testencoding, mri_trencoding, mri_valencoding, mri_testencoding, label_train, label_val, label_test, df1


def loadData(prep_df, batch_size, num_workers, sampler):
    
    return  DataLoader(
            prep_df,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )

def get_data_loaders(csv_train_path, csv_val_path, csv_test_path, tokenizer, config):

    num_workers = config['num_workers'] 
    batch_size = config['batch_size']
    seq_len = config['SEQ_LEN']

    mri_report_section = config['mri_report_section']
    pa_report_section = config['pa_report_section']

    
    if config['label'] == 'cancer_presence':
        label = 'any_cancer_1_3'
    elif config['label'] == 'cancer_status':
        label = 'cancer_status_a_e'

    df = read_data(csv_train_path)

    pa_trencoding, pa_valencoding, pa_testencoding, mri_trencoding, mri_valencoding, mri_testencoding, label_train, label_val, label_test, df1 = preprocess_data(df, tokenizer, seq_len, mri_report_section, pa_report_section, label)
     

    pa_train_seq = torch.tensor(pa_trencoding['input_ids'])
    pa_train_mask = torch.tensor(pa_trencoding['attention_mask'])
    pa_train_token_ids = torch.tensor(pa_trencoding['token_type_ids'])
    mri_train_seq = torch.tensor(mri_trencoding['input_ids'])
    mri_train_mask = torch.tensor(mri_trencoding['attention_mask'])
    mri_train_token_ids = torch.tensor(mri_trencoding['token_type_ids'])
    train_y = torch.tensor(label_train.tolist())

    pa_val_seq = torch.tensor(pa_valencoding['input_ids'])
    pa_val_mask = torch.tensor(pa_valencoding['attention_mask'])
    pa_val_token_ids = torch.tensor(pa_valencoding['token_type_ids'])
    mri_val_seq = torch.tensor(mri_valencoding['input_ids'])
    mri_val_mask = torch.tensor(mri_valencoding['attention_mask'])
    mri_val_token_ids = torch.tensor(mri_valencoding['token_type_ids'])
    val_y = torch.tensor(label_val.tolist())

    pa_test_seq = torch.tensor(pa_testencoding['input_ids'])
    pa_test_mask = torch.tensor(pa_testencoding['attention_mask'])
    pa_test_token_ids = torch.tensor(pa_testencoding['token_type_ids'])
    mri_test_seq = torch.tensor(mri_testencoding['input_ids'])
    mri_test_mask = torch.tensor(mri_testencoding['attention_mask'])
    mri_test_token_ids = torch.tensor(mri_testencoding['token_type_ids'])
    test_y = torch.tensor(label_test.tolist())

    # wrap tensors
    train_data = TensorDataset(pa_train_seq, pa_train_mask, pa_train_token_ids, mri_train_seq, mri_train_mask, mri_train_token_ids,train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # Train Data Loader
    traindata = loadData(train_data, batch_size, num_workers, train_sampler)

    # wrap tensors
    val_data = TensorDataset(pa_val_seq, pa_val_mask, pa_val_token_ids, mri_val_seq,mri_val_mask, mri_val_token_ids,val_y)
    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)
    # Val Data Loader
    valdata = loadData(val_data, batch_size, num_workers, val_sampler)

    # wrap tensors
    test_data = TensorDataset(pa_test_seq, pa_test_mask, pa_test_token_ids, mri_test_seq, mri_test_mask, mri_test_token_ids, test_y)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # Val Data Loader
    testdata = loadData(test_data, batch_size, num_workers, test_sampler)

    print('Number of data in the train set', len(traindata))
    print('Number of data in the validation set', len(valdata))
    print('Number of data in the test set', len(testdata))

    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(df_used['label'].values.tolist()), 
                                    y=df_used['label'])

    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    

    return traindata, valdata, testdata, weights

