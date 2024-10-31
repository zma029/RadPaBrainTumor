import time
from babel.dates import format_date, format_datetime, format_time
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from utils.visualization import loss_curve
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import os

# SEQ_LEN = 128
# batch_size = 2
# epochs = 2
# learning_rate = 1e-3 # Controls how large a step is taken when updating model weights during training.
# steps_per_epoch = 20
# num_workers = 0

def trainer(config, model, train_loader, val_loader, test_loader, class_wts, device, logger, save_model_name, ckpt_dir):

    class_wts = class_wts.to(device)

    # cross_entropy  = nn.CrossEntropyLoss(weight=class_wts)
    cross_entropy  = nn.CrossEntropyLoss()

    #optimizer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [{'params': [p for n, p in param_optimizer 
                                        if not any(nd in n for nd in no_decay)],'weight_decay':0.001},
                            {'params': [p for n, p in param_optimizer 
                                        if any(nd in n for nd in no_decay)],'weight_decay':0.0}]

    print('Preparing the optimizer...')
    #optimizer 
    optimizer = AdamW(optimizer_parameters, lr=config['learning_rate'])
    # steps = steps_per_epoch
    # scheduler = get_linear_schedule_with_warmup(                    
    #     optimizer,
    #     num_warmup_steps = 0,
    #     num_training_steps = steps
    # )

    # best_valid_loss = float('inf')
    best_val_f1 = 0
    best_test_f1 = 0

    # Empty lists to store training and validation loss of each epoch
    epoch_list = []
    train_losses=[]
    valid_losses=[]
    train_microf1_list = []
    train_accuracy_list = []
    valid_microf1_list = []
    valid_accuracy_list = []
    test_microf1_list = []
    test_accuracy_list = []

    # for each epoch perform training and evaluation
    for epoch in range(config['start_epoch'], config['total_epochs']):

         # log is dict
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, config['total_epochs']))
        
        # train model
        train_loss, train_accuracy, train_microf1, train_macrof1, train_f1, train_precision, train_recall \
            = train_Bert(model, train_loader, device, cross_entropy, optimizer)

        logger.info(
            f'Epoch {epoch + 1}, training is over, train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, train_microf1: {train_microf1:.3f}, '
            f'train_macrof1: {train_macrof1:.3f}, train_f1: {train_f1:.3f}, '
            f'lr: {optimizer.param_groups[0]["lr"]}')
        
        print(f'Epoch {epoch + 1}, training is over, train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, train_microf1: {train_microf1:.3f}, '
            f'train_macrof1: {train_macrof1:.3f}, train_f1: {train_f1:.3f}, lr: {optimizer.param_groups[0]["lr"]}')
        
        #evaluate model
        valid_loss, valid_accuracy, valid_microf1, valid_macrof1, valid_f1, val_precision, val_recall\
              = evaluate_Bert(model, val_loader, device, cross_entropy)

        logger.info(
            f'Epoch {epoch + 1}, validation is over, val_loss: {valid_loss:.3f}, val_accuracy: {valid_accuracy:.3f}, valid_microf1: {valid_microf1:.3f}, '
            f'valid_macrof1: {valid_macrof1:.3f}, valid_f1: {valid_f1:.3f}' 
            )
        
        print(f'Epoch {epoch + 1}, validation is over, val_loss: {valid_loss:.3f}, val_accuracy: {valid_accuracy:.3f}, valid_microf1: {valid_microf1:.3f}, '
            f'valid_macrof1: {valid_macrof1:.3f}, valid_f1: {valid_f1:.3f}'
            )
        
        print('Evaluation done for epoch {}'.format(epoch + 1))

        test_loss, test_accuracy, test_microf1, test_macrof1, test_f1, test_precision, test_recall\
              = test_Bert(model, test_loader, device, cross_entropy)

        logger.info(
            f'Epoch {epoch + 1}, testing is over, test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}, test_microf1: {test_microf1:.3f}, '
            f'test_macrof1: {test_macrof1:.3f}, test_f1: {test_f1:.3f}' 
            )
        
        print(f'Epoch {epoch + 1}, testing is over, test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}, test_microf1: {test_microf1:.3f}, '
            f'test_macrof1: {test_macrof1:.3f}, test_f1: {test_f1:.3f}'
            )
        
        #save the latest model
        print('Saving latest model...')
        # trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        # torch.save(trainable_params, os.path.join(ckpt_dir, 'latest_'+ save_model_name))
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest_'+ save_model_name))

        #save the best val model
        if valid_microf1 > best_val_f1:
            best_val_f1 = valid_microf1
            print('Saving best val model...')
            # trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
            # torch.save(trainable_params, os.path.join(ckpt_dir, 'best_val_'+ save_model_name))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_val_'+ save_model_name)) 

        #save the best test model
        if test_microf1 > best_test_f1:
            best_test_f1 = test_microf1
            print('Saving best test model...')
            # trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
            # torch.save(trainable_params, os.path.join(ckpt_dir, 'best_test_'+ save_model_name))
            
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_test_'+ save_model_name))

        # append training and validation loss
        epoch_list.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracy_list.append(train_accuracy)
        train_microf1_list.append(train_microf1)
        valid_accuracy_list.append(valid_accuracy)
        valid_microf1_list.append(valid_microf1)
        test_microf1_list.append(test_microf1)
        test_accuracy_list.append(test_accuracy)

        df = pd.DataFrame({
        'Epoch': epoch_list,
        'Train Loss': train_losses,
        'Validation Loss': valid_losses,
        'Train F1': train_microf1_list,
        'Train accuracy': train_accuracy_list,
        'Validation F1': valid_microf1_list,
        'Validation accuracy': valid_accuracy_list,
        'Test F1': test_microf1_list,
        'Test accuracy': test_accuracy_list

        })

        # 保存到 CSV
        experiment_name = save_model_name.split('.')[0]
        df.to_csv(os.path.join(config['result_dir'], f'{experiment_name}_training_results.csv'), index=False)
        # loss_curve(train_losses, valid_losses, train_f1, val_f1)

    
def train_Bert(model, train_loader, device, cross_entropy, optimizer):

    print('Training...')
    model.train()
    total_loss =  0

    total_correct = 0
    total_predictions = 0

    total_preds=[]
    total_labels = []


    for step, batch in enumerate(tqdm(train_loader, desc="Training")):

    
        # if step % 50 == 0 and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))

        if torch.cuda.is_available():
            batch = [r.to(device) for r in batch]

        pa_id, pa_mask, pa_token_type_ids, mri_id2, mri_mask2, mri_token_type_ids2, labels = batch
        # clear previously calculated gradients 
        optimizer.zero_grad()        
        # get model predictions for the current batch
        preds = model(pa_id, pa_mask, pa_token_type_ids, mri_id2, mri_mask2, mri_token_type_ids2)
        labels = labels - 1
        
        labels = labels.long()
        loss = cross_entropy(preds, labels)
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
  
        total_loss = total_loss + loss.item()

        _, predicted = torch.max(preds, 1)
        total_predictions += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        total_preds.extend(predicted.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_predictions

    average_f1 = f1_score(total_labels, total_preds, average='weighted')
    micro_f1 = f1_score(total_labels, total_preds, average='micro')
    macro_f1 = f1_score(total_labels, total_preds, average='macro')

    # 计算 precision 和 recall
    precision = precision_score(total_labels, total_preds, average='micro')
    recall = recall_score(total_labels, total_preds, average='micro')
    
    #returns the loss and predictions
    return avg_loss, avg_accuracy, micro_f1, macro_f1, average_f1, precision, recall


def evaluate_Bert(model, val_loader, device, cross_entropy):
  
    print("\nEvaluating...")
    t0 = time.time()
    
    model.eval() # deactivate dropout layers
    total_loss = 0
    
    # empty list to save the model predictions
    total_correct = 0
    total_predictions = 0
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(val_loader):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_loader)))

        if torch.cuda.is_available():
            # push the batch to gpu
            batch = [t.to(device) for t in batch]

        sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2, labels = batch

        # deactivate autograd
        with torch.no_grad(): # Dont store any previous computations, thus freeing GPU space

            # model predictions
            preds = model(sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2)
            labels = labels - 1
            # compute the validation loss between actual and predicted values
            labels = labels.long()
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()

            _, predicted = torch.max(preds, 1)
            total_predictions += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            total_preds.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
        

        torch.cuda.empty_cache()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_loader) 
    avg_accuracy = total_correct / total_predictions

    average_f1 = f1_score(total_labels, total_preds, average='weighted')
    micro_f1 = f1_score(total_labels, total_preds, average='micro')
    macro_f1 = f1_score(total_labels, total_preds, average='macro')

    # 计算 precision 和 recall
    precision = precision_score(total_labels, total_preds, average='micro')
    recall = recall_score(total_labels, total_preds, average='micro')

    return avg_loss, avg_accuracy, micro_f1, macro_f1, average_f1, precision, recall

def test_Bert(model, test_loader, device, cross_entropy):
  
    print("\nTesting...")
    t0 = time.time()
    
    model.eval() # deactivate dropout layers
    total_loss = 0
    
    # empty list to save the model predictions
    total_correct = 0
    total_predictions = 0
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(test_loader):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_loader)))

        if torch.cuda.is_available():
            # push the batch to gpu
            batch = [t.to(device) for t in batch]

        sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2, labels = batch

        # deactivate autograd
        with torch.no_grad(): # Dont store any previous computations, thus freeing GPU space

            # model predictions
            preds = model(sent_id, mask, token_type_ids, sent_id2, mask2, token_type_ids2)
            labels = labels - 1
            # compute the validation loss between actual and predicted values
            labels = labels.long()
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()

            _, predicted = torch.max(preds, 1)
            total_predictions += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            total_preds.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
        

        torch.cuda.empty_cache()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(test_loader) 
    avg_accuracy = total_correct / total_predictions

    average_f1 = f1_score(total_labels, total_preds, average='weighted')
    micro_f1 = f1_score(total_labels, total_preds, average='micro')
    macro_f1 = f1_score(total_labels, total_preds, average='macro')

    # 计算 precision 和 recall
    precision = precision_score(total_labels, total_preds, average='micro')
    recall = recall_score(total_labels, total_preds, average='micro')

    return avg_loss, avg_accuracy, micro_f1, macro_f1, average_f1, precision, recall



