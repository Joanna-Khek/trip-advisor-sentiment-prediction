import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay

def compute_accuracy(model, data_loader, device):

    train_accuracy = Accuracy(task='BINARY').to(device)

    with torch.no_grad():

        for batch_idx, batch in enumerate(data_loader):

            ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']

            _, predicted_labels = torch.max(logits, 1)

            train_accuracy(predicted_labels, labels)

    # total train accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()
    train_accuracy.reset()

    return total_train_accuracy

def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []

    with torch.no_grad():

        for batch_idx, batch in enumerate(data_loader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            _, predicted_labels = torch.max(logits, 1)

            # save all the targets and predictions
            all_targets.extend(labels.to('cpu').numpy())
            all_predictions.extend(predicted_labels.to('cpu').numpy())
    
    cm = confusion_matrix(all_predictions, all_targets)
    print(classification_report(all_predictions, all_targets))
    print(ConfusionMatrixDisplay(cm).plot())


def show_misclassified(model, data_loader, device, tokenizer):
    
    incorrect_text = []
    true_label = []
    pred_label = []
    prob = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(data_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            _, predicted_labels = torch.max(logits, 1)
            
            # If labels do not match predicted labels via index
            unmatched = torch.where(labels != predicted_labels)[0]

            for idx in unmatched:
                #incorrect_text.append(tokenizer.convert_ids_to_tokens(input_ids[idx], skip_special_tokens=True))
                incorrect_text.append(tokenizer.decode(input_ids[idx], skip_special_tokens=True))
                true_label.append(labels[idx].item())
                pred_label.append(predicted_labels[idx].item())
                prob.append(F.softmax(logits[idx], dim=0).to("cpu"))

    # Save into dataframe
    df_error = pd.DataFrame()
    df_error['text'] = incorrect_text
    df_error['true_label'] = true_label
    df_error['pred_label'] = pred_label
    df_error['probability'] = prob

    return df_error

def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch,
                       results_dir=None, averaging_iterations=100):

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss_list)),
             (minibatch_loss_list), label='Minibatch Loss')

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([
            0, np.max(minibatch_loss_list[1000:])*1.5
            ])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        plt.savefig(image_path)

def plot_accuracy(train_acc_list, valid_acc_list, results_dir):

    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs+1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(
            results_dir, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)