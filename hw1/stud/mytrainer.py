# let's start with all the imports
# NOTE: part of this code is taken from notebook #5
import torch
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from myvocab import Vocab
from typing import Dict


class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(self, model:nn.Module, loss_function, optimizer, label_vocab:Vocab, device):
        """Constructor of our trainer
        Args:
            model (nn.Module): model to train
            loss_function (nn.Loss): loss function to use
            optimizer (nn.Optim): optimizer to use
            label_vocab (Vocab): label vocabulary used to decode the output
            device (torch.device): device where to perform training and validation
        """
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        
        self.predictions = [] # will contain validation set predictions, useful to plot confusion matrix and a report after training
        self.truths = [] # will contain validation set truths, useful to plot confusion matrix and a report after training

    def train(self, train_data:DataLoader, valid_data:DataLoader, epochs:int, patience:int, path:str) -> Dict[str, list]:
        """Train and validate the model with early stopping with patience for the given number of epochs
        Args:
            train_data (DataLoader): a DataLoader instance containing the training dataset
            valid_data (DataLoader): a DataLoader instance used to evaluate learning progress
            epochs: the number of times to iterate over train_data
            patience (int): patience for early stopping
            path (str): path where to save weights of best epoch
        Returns:
            Dict[str, list]: dictionary containing mappings { metric:value }
        """

        train_history = []
        valid_loss_history = []
        valid_f1_history = []
        patience_counter = 0
        best_f1 = 0.0

        print('Training on', self.device, 'device')
        print('Start training ...')
        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train() # put model in train mode

            for batch in tqdm(train_data, leave=False):
                x_data, pos_data, y_data = batch['x'].to(self.device), batch['pos'], batch['y'].to(self.device)
                if pos_data is not None: pos_data = pos_data.to(self.device) # if using pos, move to device

                self.optimizer.zero_grad()
                logits = self.model(x_data, pos_data) # forward step output has shape: batchsize, max_seq, 13 classes

                if self.model.crf is not None: # if using crf, then use its loss computation to optimize the model
                    mask = (y_data != -100)
                    batch_losses = self.model.crf.forward(logits, y_data.where(mask, torch.LongTensor([0]).to(self.device)), mask)
                    # I mapped the -100 "padding" labels to 0 just because TorchCRF doesn't want indices out of the
                    # number of classes, but anyway they will be ignored thanks to the mask parameter
                    sample_loss = - batch_losses.mean()
                else: # otherwise, use standard given loss function on logits
                    sample_loss = self.loss_function(logits.permute(0,2,1), y_data) # permute to match nn.CrossEntropyLoss input dims
                
                sample_loss.backward() # backpropagation
                self.optimizer.step() # optimize parameters

                epoch_loss += sample_loss.item() * x_data.shape[0] # avg batch loss * precise number of batch elements

            avg_epoch_loss = epoch_loss / len(train_data.dataset) # total loss / number of samples = average sample loss for this epoch
            train_history.append(avg_epoch_loss)
            print('  [E:{:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_metrics = self.evaluate(valid_data) # validation step
            valid_loss_history.append(valid_metrics["loss"])
            valid_f1_history.append(valid_metrics["f1"])
            print('\t[E:{:2d}] valid f1 = {:0.4f}%'.format(epoch, valid_metrics["f1"]*100))

            # save model if the validation metric is the best ever
            if valid_metrics["f1"] > best_f1: 
                best_f1 = valid_metrics["f1"]
                torch.save(self.model.state_dict(), path)

            stop = epoch > 0 and valid_f1_history[-1] < valid_f1_history[-2]  # check if early stopping
            if stop:
                patience_counter += 1
                if patience_counter > patience: # in case we exhausted the patience, we stop
                    print('\tEarly stop\n')
                    break
                else:
                    print('\t-- Patience')
            print()

        print('Done!')
        return {
            'train_history': train_history,
            'valid_loss_history': valid_loss_history,
            'valid_f1_history': valid_f1_history
        }
    
    @torch.no_grad()
    def compute_f1(self, input:torch.Tensor, truth:torch.Tensor, given_logits:bool=True) -> float:
        """Evaluation function to compute f1 score for a batch of predictions
        Args:
            input (torch.Tensor): logits produced by the model or predicted indices produced by the crf layer
            truth (torch.Tensor): ground truths in encoded format
            given_logits (bool, optional): True if the input parameter are raw logits, False if input are predictions. Defaults to True.
        Returns:
            float: the f1 score for the given batch
        """
        list_predictions = []
        list_truth = []

        if not given_logits:
            list_predictions = [[self.label_vocab.i2w[index] for index in sentence] for sentence in input] # convert back from indices to labels
            list_truth = [[self.label_vocab.i2w[index] for index in sentence if index != -100] for sentence in truth.tolist()]

        else:
            torch_predictions = torch.argmax(input, -1) # torch_predictions and truth shapes now match: batch, max_seq

            for i in range(truth.shape[0]):
                pred, gt = torch_predictions[i], truth[i]
                mask = (gt != -100) # to discard the padding in this sentence
                list_predictions.append([self.label_vocab.i2w[index] for index in pred[mask].tolist()]) # convert back from index to label
                list_truth.append([self.label_vocab.i2w[index] for index in gt[mask].tolist()])
        
        assert len(list_predictions) == len(list_truth)
        self.predictions.extend(list_predictions) # add predictions and truths to the global list
        self.truths.extend(list_truth)
        return f1_score(list_truth, list_predictions, average="macro")

    def evaluate(self, valid_data:DataLoader) -> Dict[str, float]:
        """ perform validation of the model
        Args:
            valid_dataset: the DataLoader to use to evaluate the model.
        Returns:
            Dict[str, float]: dictionary containing mappings { metric:value }
        """
        valid_loss = 0.0
        f1 = 0.0
        self.predictions = [] # reset predictions and truths lists
        self.truths = []

        self.model.eval() # inference mode
        with torch.no_grad():
            for batch in tqdm(valid_data, leave=False):
                x_data, pos_data, y_data = batch['x'].to(self.device), batch['pos'], batch['y'].to(self.device)
                if pos_data is not None: pos_data = pos_data.to(self.device) # if using pos, move to device

                logits = self.model(x_data, pos_data)
                
                if self.model.crf is not None: # if using crf, then use its loss computation to evaluate the model
                    mask = (y_data != -100)
                    batch_losses = self.model.crf.forward(logits, y_data.where(mask, torch.LongTensor([0]).to(self.device)), mask)
                    sample_loss = - batch_losses.mean()
                    f1 += self.compute_f1(self.model.crf.viterbi_decode(logits, mask), y_data, given_logits=False)

                else: # otherwise, use standard given loss function on logits
                    sample_loss = self.loss_function(logits.permute(0,2,1), y_data) # permute to match CrossEntropyLoss input dim
                    f1 += self.compute_f1(logits, y_data)

                valid_loss += sample_loss.item() * x_data.shape[0] # avg batch loss * precise number of batch elements
        
        return {
            "loss": valid_loss / len(valid_data.dataset), # total loss / number of samples = average sample loss for validation step
            "f1": f1/len(valid_data) # total f1 / number of batches = average batch f1 for this validation step
        }
 
    def generate_cm(self, path:str):
        """plot a classification report and then save to image the confusion matrix on the validation set of this trainer
        Args:
            path (str): path where to save the image
        """

        print(classification_report(self.truths, self.predictions)) # print a classification report

        labels = ["B-CORP", "I-CORP", "B-PROD", "I-PROD", "B-CW", "I-CW",
                "B-LOC", "I-LOC", "B-PER", "I-PER", "B-GRP", "I-GRP", "O"]
        cm = np.around(confusion_matrix([label for sentence in self.truths for label in sentence],
                                        [label for sentence in self.predictions for label in sentence],
                                        labels=labels,
                                        normalize="true"),
                        decimals=2) # normalize over ground truths

        df_cm = pd.DataFrame(cm, index=labels, columns=labels) # create a dataframe just for easy plotting with seaborn
        plt.figure(figsize = (9,8))
        cm_plot = sn.heatmap(df_cm, annot=True, fmt='g')
        cm_plot.set_xlabel('Predicted labels') # add some interpretability
        cm_plot.set_ylabel('True labels')
        cm_plot.set_title('Confusion Matrix')
        cm_plot.figure.savefig(path)
        return

    @staticmethod
    def plot_logs(logs:Dict[str, list], path:str):
        """Utility function to generate plot for metrics of loss vs validation. Code is taken from notebook #5
        Args:
            logs (Dict[str, list]): dictionary containing the metrics
            path (str): path of the image to be saved
        """
        plt.figure(figsize=(8,6)) # create the figure

        # plot losses over epochs
        plt.plot(list(range(len(logs['train_history']))), logs['train_history'], label='Train loss')
        plt.plot(list(range(len(logs['valid_loss_history']))), logs['valid_loss_history'], label='Validation loss')

        # add some labels
        plt.title("Train vs Validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(path, bbox_inches='tight')
        return

    