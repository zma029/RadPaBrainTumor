import torch
from torch.utils.data import DataLoader



class TrainDataLoader(DataLoader):
    def __init__(self, config, tokenizer, split, shuffle):
        self.args = config
        self.csv_path = config[f'csv_{split}_path']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        
        self.dataset = GLBTrainDataset(self.args, self.tokenizer, self.csv_path)
    
        self.cls_weight = self.dataset.cls_weights
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(batch):

        pa_train_seq, pa_train_mask, pa_train_token_ids, mri_train_seq, mri_train_mask, mri_train_token_ids,label = zip(*batch)


        return pa_train_seq, pa_train_mask, pa_train_token_ids, mri_train_seq, mri_train_mask, mri_train_token_ids,label