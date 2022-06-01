# let's start with all the imports
# NOTE: part of this code is taken from notebook #6 - POS tagging
import torch
from torch import nn
from TorchCRF import CRF
from stud.myvocab import Vocab


class ModHParams():
    """A wrapper class that contains the hyperparamers of the model"""
    
    def __init__(self,
                words_vocab:Vocab,
                labels_vocab:Vocab,
                word_embedding_dim:int=300,
                word_embeddings:nn.Embedding=None,
                hidden_dim:int=256,
                lstm_layers:int=3,
                dropout:float=0.35,
                pos_embedding_dim:int=25,
                fc_layers:int=1,
                use_crf:bool=True):
        """NER model's hyperparameters initialization
        Args:
            words_vocab (Vocab): vocabulary of words
            labels_vocab (Vocab): vocabulary of output labels
            word_embedding_dim (int, optional): word embeddings dimensions. Defaults to 300.
            word_embeddings (nn.Embedding, optional): pretrained word embeddings layer to load into the model. Defaults to None.
            hidden_dim (int, optional): hidden state's dimension of the LSTM. Defaults to 256.
            lstm_layers (int, optional): number of LSTM layers. Defaults to 3.
            dropout (float, optional): dropout value to apply. Defaults to 0.35.
            pos_embedding_dim (int, optional): pos embeddings dimensions. Defaults to 25.
            fc_layers (int, optional): number of fully connected layers after the LSTM. Defaults to 1.
            use_crf (bool, optional): whether to use a crf on top of the classification layer. Defaults to True.
        """
        self.vocab_size = len(words_vocab) + 1 # +1 for padding 
        self.hidden_dim = hidden_dim
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.num_classes = len(labels_vocab) # number of different NE tags
        self.bidirectional = True
        self.lstm_layers = lstm_layers
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.word_embeddings = word_embeddings
        self.crf = use_crf


class NERModel(nn.Module):
    """My model to perform NER"""

    def __init__(self, hparams):
        """constructor of the model
        Args:
            hparams: an object embedding all the hyperparameters
        """
        super(NERModel, self).__init__()

        #word embeddings layer
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.word_embedding_dim, padding_idx=0)
        if hparams.word_embeddings is not None: # initialize embedding layer from pretrained ones
            self.word_embedding = hparams.word_embeddings
        
        #pos embeddings layer
        self.pos_embedding = None
        if hparams.pos_embedding_dim != 0:
            self.pos_embedding = nn.Embedding(18, hparams.pos_embedding_dim, padding_idx=0) # 18 = 17 upos tags + 1 for padding

        #LSTM layer
        self.lstm = nn.LSTM(hparams.word_embedding_dim + hparams.pos_embedding_dim, # in forward method I will concatenate the two embeddings
                            hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.lstm_layers, 
                            batch_first=True,
                            dropout = hparams.dropout if hparams.lstm_layers > 1 else 0)

        # dropout layer to allow some regularization
        self.dropout = nn.Dropout(hparams.dropout)

        # compute lstm output dim to create the linear layers
        lstm_output_dim = hparams.hidden_dim if not hparams.bidirectional else hparams.hidden_dim * 2
        # feed forward layers before classification
        modules = []
        for _ in range(1, hparams.fc_layers): # iteratively add fc layers before the classification one
            modules.append(nn.Linear(lstm_output_dim, lstm_output_dim//2)) # halving dimension at each layer
            modules.append(nn.ReLU)
            modules.append(nn.Dropout(hparams.dropout))
            lstm_output_dim = lstm_output_dim//2

        self.fc = nn.Sequential(*modules)

        # last fc layer for classification
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

        # conditional random field on top of the classification layer, due to the behaviour of TorchCRF library this layer returns per-sample losses,
        # so I will use it outside of the model class (i.e. in the train/eval/predict loop) but instantiate here for logical cohesion with other layers
        self.crf = CRF(hparams.num_classes) if hparams.crf else None
    
    def forward(self, x:torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        """forward method of the model
        Args:
            x (torch.Tensor): word indices of the sentence, with padding
            pos (torch.Tensor): pos indices of the sentence, with padding

        Returns:
            torch.Tensor: logits for each named entity class, for each sentence in the batch
        """
        embeddings = self.word_embedding(x)
        if pos is not None:
            p_embeddings = self.pos_embedding(pos)
            embeddings = torch.cat([embeddings, p_embeddings], dim=2) # concatenate word and pos embeddings
            
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        o = self.fc(o)
        output = self.classifier(o)
        return output
