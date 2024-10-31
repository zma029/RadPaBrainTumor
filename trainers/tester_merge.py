import pandas as pd
import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch.nn.functional import softmax
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


def test_model(model, test_loader, device, config):

    print('Start testing...')
    model.to(device)
    model.eval()
    
    total_correct = 0
    total_predictions = 0
    
    test_preds = []
    test_labels = []
    test_scores = []
    torch.set_grad_enabled(False)

    with torch.no_grad():
        for j, test_batch in enumerate(test_loader):

            inference_status = 'Batch ' + str(j + 1)

            print(inference_status, end='\r')

            if torch.cuda.is_available():
                batch = [r.to(device) for r in test_batch]
            
            sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2, labels = batch
            labels = labels - 1
            scores = model(sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2)
            probabilities = softmax(scores, dim=1)

            _, predicted = torch.max(scores, 1)
            total_predictions += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
            test_scores.extend(probabilities.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    avg_accuracy = total_correct / total_predictions

    average_f1 = f1_score(test_labels, test_preds, average='weighted')
    micro_f1 = f1_score(test_labels, test_preds, average='micro')
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    total_f1 = f1_score(test_labels, test_preds, average=None)
    # 计算 precision 和 recall
    precision = precision_score(test_labels, test_preds, average='micro')
    total_precision = precision_score(test_labels, test_preds, average=None)
    recall = recall_score(test_labels, test_preds, average='micro')
    total_recall = recall_score(test_labels, test_preds, average=None)

    # 二分类
    # report = classification_report(test_labels, test_preds, target_names=['Class 0', 'Class 1'])
    # precision, recall, fscore, support = precision_recall_fscore_support(test_labels, test_preds)

    # 多分类
    report = classification_report(test_labels, test_preds, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])

    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average F1 Score: {average_f1}')
    print(f'Micro F1 Score: {micro_f1}')
    print(f'Macro F1 Score: {macro_f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    ################################  绘制confusion matrix，计算每一类的accuracy
    # 统计TN FP FN TP
    # conf_matrix = confusion_matrix(test_labels, test_preds)
    # TN, FP, FN, TP = conf_matrix.ravel()

    # target_names = ['true', 'predicted']

    data = {'true': test_labels,
        'predicted': test_preds}

    df_pred_BERT = pd.DataFrame(data)

    # df_cleaned = df_pred_BERT.loc[~((df_pred_BERT == 4)| (df_pred_BERT == 3)).any(axis=1)]

    # df_cleaned = df_pred_BERT.loc[~((df_pred_BERT == 2)).any(axis=1)]

    labels = sorted(df_pred_BERT['true'].unique())
   
    confusion_matrix = pd.crosstab(df_pred_BERT['true'], df_pred_BERT['predicted'], rownames=['True'], colnames=['Predicted'],dropna=False)
    confusion_matrix = confusion_matrix.reindex(index=labels, columns=labels, fill_value=0)

    confusion_matrix_percentage = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100

    # confusion_matrix_renamed = confusion_matrix_percentage.rename(index={0: 'Improving', 1: 'Worsening', 2:'Stable'},
    #                                                columns={0: 'Improving', 1: 'Worsening', 2: 'Stable'})
    
    # # confusion_matrix_renamed = confusion_matrix_percentage.rename(index={0: 'Tumor Presence', 1: 'No Tumor Presence'},
    # #                                                columns={0: 'Tumor Presence', 1: 'No Tumor Presence'})
    
    # confusion_matrix_percentage = confusion_matrix_percentage.applymap(lambda x: f'{x:.2f}%')

    # plt.figure()
    # sns.heatmap(confusion_matrix_renamed, annot=confusion_matrix_percentage, fmt="", cmap='Blues')
    # plt.savefig('confusion_matrix_merge_3.png', dpi=300, bbox_inches='tight')

       # annotations = np.empty_like(confusion_matrix_percentage)

    # # 填充对角线上的百分比值，其余位置显示原始值
    # for i in range(confusion_matrix.shape[0]):
    #     for j in range(confusion_matrix.shape[1]):
    #         if i == j:
    #             annotations[i, j] = f'{confusion_matrix_percentage.iloc[i, j]:.2f}%'
     
    

    # # ################################################# 计算 AUC 和 绘制 ROC 曲线
    # 把标签和预测概率存起来

    y_scores = np.array(test_scores)
    y_true = np.array(test_labels)

    data = {'true': test_labels,
        'predicted': test_scores}

    df_pred_BERT = pd.DataFrame(data)

    df_pred_BERT.to_csv(f'{config["result_dir"]}/test_result_{config["label"]}.csv', index=False)
    
    # micro-averaged AUC

    y_true_bin = label_binarize(test_labels, classes=[0, 1, 2])
    # 二分类  
    # y_scores = y_scores[:,1]

    # 多分类
    auc_score = roc_auc_score(y_true_bin, y_scores, average='micro')
    print(f'Microaveraged AUC: {auc_score}')

    # micro-averaged AUC 的置信区间

    n_bootstraps = 1000
    rng = np.random.RandomState(42)  # 固定随机种子以复现结果
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # 创建重抽样的索引，并带替换地抽取样本
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < len(np.unique(y_true)):
            continue  # 确保重抽样中包含所有类别

        # 计算每个bootstrap样本的AUC
        score = roc_auc_score(y_true_bin[indices], y_scores[indices], average='micro')
        bootstrapped_scores.append(score)

    alpha = 0.95
    sorted_scores = np.sort(bootstrapped_scores)
    confidence_lower = sorted_scores[int((1.0-alpha)/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0+alpha)/2 * len(sorted_scores))]

    print(f'Microaveraged AUC: {np.mean(bootstrapped_scores)}')
    print(f'{int(alpha*100)}% confidence interval for the AUC: [{confidence_lower:.3f}, {confidence_upper:.3f}]')



    # # # # # # # # # # # # # # # # # # # # # 为每个类别计算ROC曲线和AUC，以及Micro-Averaged ROC曲线
    # plt.figure()
    # lw = 2
    # if len(np.unique(test_labels)) == 2:
    #     plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    # else:
    #     plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc["micro"]:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()


   
    # plt.show()

   

    # print(classification_report(test_labels, test_preds, target_names=target_names))

    





