import matplotlib.pyplot as plt
import seaborn as sns

def loss_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')

    plt.tight_layout()
    
    plt.savefig('results/loss_curve.png')




def confusion_matrix(gen_labels_test_set, best_model_preds, test_dataset):
    cm = confusion_matrix(gen_labels_test_set, best_model_preds)
    labels = set([label[2] for label in test_dataset])
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label', labelpad=20, fontsize=14)
    plt.ylabel('Truth Label', labelpad=20, fontsize=14)
    plt.show()