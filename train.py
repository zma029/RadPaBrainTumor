import warnings 
warnings.filterwarnings("ignore")
from utils.util import SetLogger
from utils.util import load_config
from models.train_model import ClinicalBertModel
import torch
from babel.dates import format_date, format_datetime, format_time
from transformers import BertTokenizer, AutoTokenizer
from data.dataset_unimodal import get_data_loaders
from trainers.trainer import trainer
from trainers.tester import test_model
from utils.util import setup_seed
import os

def main():

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    modality_name = "unimodal_determine"

    config = load_config('options/config.json')
    
    setup_seed(config["seed"])

    logger = SetLogger(f'{config["result_dir"]}/train.log', 'a')

    # -------------------------------
    # save the config
    params = ''
    for key, value in config.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    print(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=True)
    tokenizer= AutoTokenizer.from_pretrained('UFNLP/gatortron-base')
    train_loader, val_loader, test_loader, cls_weights= get_data_loaders(tokenizer, config)

    print(f"train_data is {len(train_loader.dataset) if train_loader is not None else 'None'}, "
            f"val_data is {len(val_loader.dataset) if val_loader is not None else 'None'} "
            )
    logger.info(f"train_data is {len(train_loader.dataset) if train_loader is not None else 'None'}, "
                f"val_data is {len(val_loader.dataset) if val_loader is not None else 'None'}"
            )
    

    print('Downloading custom model...')
   
    model = ClinicalBertModel(n_classes=config['num_classes'], model_name=config['model_name']).to(device)
    
    print(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    logger.info(f'finish instantiate model!, Trainable parameters:{str(model).split("Trainable parameters:")[1]}M')
    model_name = config['model_name'].split('/')[-1]
    save_model_name = f"no_att_no_cls_w_{model_name}_{config['SEQ_LEN']}_cls_{config['num_classes']}_pa_{config['pa_report_section']}.pt"
    ckpt_dir = os.path.join(config['ckpt_dir'], modality_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # save_model_name = 'checkpoints/unimodal/best_test_no_att_no_cls_w_gatortron-base_512_cls_5_mri_report.pt'
    # trainable_params = torch.load(f'{save_model_name}')
        
    # model.load_state_dict(trainable_params, strict=False)
    
    # continue_train = True
    # if continue_train:
    #     trainable_params = torch.load(f'latest_{save_model_name}')
        
    #     model.load_state_dict(trainable_params, strict=False)

    trainer(config, model, train_loader, val_loader, test_loader, cls_weights, device, logger, save_model_name, ckpt_dir)



if __name__ == "__main__":
    main()