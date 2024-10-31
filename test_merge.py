import warnings 
warnings.filterwarnings("ignore")
from utils.util import SetLogger
from utils.util import load_config
import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer
from models.train_model_merge import ClinicalBertModel
from utils.util import setup_seed
from data.dataset_test_merge import get_data_loaders
from trainers.tester_merge import test_model


def main():

    config = load_config('options/config.json')

    logger = SetLogger(f'{config["result_dir"]}/test.log', 'a')

    setup_seed(config["seed"])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading BERT tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
    test_loader = get_data_loaders(tokenizer, config)

    print('Downloading the BERT custom model...')
   
    model = ClinicalBertModel(n_classes=config['num_classes'], model_name=config['model_name']).to(device)

    # model_name = config['model_name'].split('/')[-1]
    # save_model_name = f"merged_no_att_no_cls_w_{model_name}_{config['SEQ_LEN']}_cls_{config['num_classes']}_mri_{config['mri_report_section']}_pa_{config['pa_report_section']}.pt"
    save_model_name = f'checkpoints/multimodal_determine/best_test_no_att_no_cls_w_gatortron-base_512_cls_{config["num_classes"]}_mri_report_pa_Comment.pt'
    trainable_params = torch.load(f'{save_model_name}')
        
    model.load_state_dict(trainable_params)
    
    test_model(model, test_loader, device, config)


if __name__ == "__main__":
    main()