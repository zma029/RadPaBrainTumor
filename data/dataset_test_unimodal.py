import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
import numpy as np


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
    return_attention_mask=True,
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
    
    batch_size = config['batch_size']
    seq_len = config['SEQ_LEN']
    num_workers = config['num_workers']

    mri_report_section = config['mri_report_section']

    pa_report_section = config['pa_report_section']
    

    if config['label'] == 'cancer_presence':
        label = 'any_cancer_1_3'
    elif config['label'] == 'cancer_status':
        label = 'cancer_status_a_e'

    df_test = pd.read_csv(config['csv_test_path'])

    testencoding, label_test = preprocess_data(df_test, tokenizer, seq_len, mri_report_section, label)


    test_seq = torch.tensor(testencoding['input_ids'])
    test_mask = torch.tensor(testencoding['attention_mask'])
    test_token_ids = torch.tensor(testencoding['token_type_ids'])
    test_y = torch.tensor(label_test.to_list())

    # wrap tensors
    test_data = TensorDataset(test_seq, test_mask, test_token_ids, test_y)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)
    # Val Data Loader
    testdata = loadData(test_data, batch_size, num_workers, test_sampler)

    print('Number of data in the test set', len(testdata))


    return testdata

