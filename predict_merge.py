import warnings 
warnings.filterwarnings("ignore")
from utils.util import SetLogger
from utils.util import load_config
import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer
from models.train_model_merge import ClinicalBertModel
from utils.util import setup_seed
from data.dataset_predict_merge import get_data_loaders
from trainers.predictor import predict_model
import os


def main():

   
    config = load_config('options/config.json')
    setup_seed(config["seed"])
    modality_name = "multimodal_determine"

    config["csv_predict_path"] =  "datasets/GLB_mri_patho_merge.csv"
    config["batch_size"] =  1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading BERT tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
    test_loader, df, study_id_to_index, index_to_study_id = get_data_loaders(tokenizer, config, config["csv_predict_path"])
    
    print('Downloading the BERT custom model...')
   
    model = ClinicalBertModel(n_classes=config['num_classes'], model_name=config['model_name']).to(device)

    ckpt_dir = os.path.join(config['ckpt_dir'], modality_name)
    model_name = config['model_name'].split('/')[-1]

    experiment_name = f"no_att_no_cls_w_{model_name}_{config['SEQ_LEN']}_cls_{config['num_classes']}_mri_{config['mri_report_section']}_pa_{config['pa_report_section']}.pt"

    save_model_name = os.path.join(ckpt_dir, 'best_test_'+ experiment_name)
    model.load_state_dict(torch.load(save_model_name))
    
    predict_model(model, test_loader, df, device, index_to_study_id, config, experiment_name)


if __name__ == "__main__":
    main()