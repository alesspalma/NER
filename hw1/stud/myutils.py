# let's start with all the imports
# NOTE: part of this code is taken from notebook #8 - Q&A
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
from myvocab import Vocab
from typing import List, Dict

# just defining a couple of utility functions

def prepare_batch(batch:List[Dict]) -> Dict[str,torch.Tensor]:
    """collate_fn for the train and dev DataLoaders, applies padding to data
    Args:
        batch (List[Dict]): a list of dictionaries, each dict is a sample from the Dataset
    Returns:
        Dict[str,torch.Tensor]: a batch into a dictionary {x:data, y:labels}
    """
    # extract features and labels from batch
    x = [sample["x"] for sample in batch]
    y = [sample["y"] for sample in batch]
    pos = None
    if "pos" in batch[0]: # if using pos tags
        pos = [sample["pos"] for sample in batch]

    # convert features to tensor and pad them
    x = pad_sequence(
            [torch.as_tensor(sample) for sample in x],
            batch_first=True
            )
    # convert and pad labels too
    y = pad_sequence(
            [torch.as_tensor(sample) for sample in y],
            batch_first=True,
            padding_value=-100
            )
    # eventually also pos tags
    if pos is not None:
        pos = pad_sequence(
            [torch.as_tensor(sample) for sample in pos],
            batch_first=True
            )

    return {"x": x, "pos": pos, "y": y}

def load_pretrained_embeddings(weights:KeyedVectors, words_vocab:Vocab, freeze:bool) -> nn.Embedding:
    """Creates the pretrained embedding layer, according to the index mapping we have in our vocabulary
    Args:
        weights (KeyedVectors): pretrained embeddings from gensim
        words_vocab (Vocab): our vocabulary of words
        freeze (bool): whether to allow fine-tuning of pretrained embeddings or not
    Returns:
        nn.Embedding: the PyTorch embedding layer
    """
    vectors = weights.vectors
    to_be_filled = np.random.randn(len(words_vocab)+1, vectors.shape[1]) # +1 for padding
    to_be_filled[0] = np.zeros(vectors.shape[1]) # zero vector for padding
    to_be_filled[1] = np.mean(vectors, axis=0) # mean vector for unknown tokens

    initialised = 0 # just for stats
    for w, i in words_vocab.w2i.items():
        if w in weights and w != "<unk>": # if the word is in the pretrained embeddings
            initialised += 1
            vec = weights[w]
            to_be_filled[i] = vec # insert in right position
        
    print("initialised embeddings: {}".format(initialised))
    print("randomly initialised embeddings: {} ".format(len(words_vocab) - initialised - 1))
    return nn.Embedding.from_pretrained(torch.FloatTensor(to_be_filled), padding_idx=0, freeze=freeze)