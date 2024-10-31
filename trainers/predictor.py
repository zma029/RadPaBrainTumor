import pandas as pd
import torch
import torch.nn as nn
import os
from torch.nn.functional import softmax

def predict_model(model, test_loader, df, device, index_to_study_id, config, experiment_name):

    print('Start testing...')
    model.to(device)
    model.eval()

    test_preds_labels = []
    test_preds_prob0 = []
    test_preds_prob1 = []
    test_preds_prob2 = []
    # test_preds_prob3 = []
    # test_preds_prob4 = []
    
    study_ids = []
    torch.set_grad_enabled(False)

    with torch.no_grad():
        for j, test_batch in enumerate(test_loader):

            inference_status = 'Batch ' + str(j + 1)

            print(inference_status, end='\r')

            if torch.cuda.is_available():
                batch = [r.to(device) for r in test_batch]

            sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2, study_id = batch
            scores = model(sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2)
            probabilities = softmax(scores, dim=1)
    
            _, predicted = torch.max(scores, 1)

            test_preds_prob0.extend(probabilities[:, 0].cpu().numpy())
            test_preds_prob1.extend(probabilities[:, 1].cpu().numpy())
            # test_preds_prob2.extend(probabilities[:, 2].cpu().numpy())
            # test_preds_prob3.extend(probabilities[:, 3].cpu().numpy())
            # test_preds_prob4.extend(probabilities[:, 4].cpu().numpy())
            test_preds_labels.extend(predicted.cpu().numpy())
            study_ids.append(study_id.cpu())

    study_ids_str = [index_to_study_id[idx.item()] for idx in study_ids]

    data = {'study_id': study_ids_str,
            'predicted_labels': test_preds_labels,
            'prob_presence': test_preds_prob0,
            'prob_no_presence':test_preds_prob1}
            # 'prob_indeterminate':test_preds_prob2}
    
    # data = {'study_id': study_ids_str,
    #         'predicted_labels': test_preds_labels,
    #         'prob_improving': test_preds_prob0,
    #         'prob_worsening':test_preds_prob1,
    #         'prob_stable':test_preds_prob2}
            # 'prob_mixed':test_preds_prob3,
            # 'prob_indeterminate':test_preds_prob4}
    
    preds_df = pd.DataFrame(data)

    combined_df = pd.merge(df, preds_df, on='study_id', how='left')
    
    savename = experiment_name.split('.')[0]
    combined_df.to_csv(os.path.join(config['result_dir'], f'glb_{savename}_prediction.csv'), index=False)



    





