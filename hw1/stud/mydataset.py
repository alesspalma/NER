# let's start with all the imports
# NOTE: part of this code is taken from notebook #6 - POS tagging
import torch
import stanza
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
from stud.myvocab import Vocab


class NERDataset(Dataset):
    """My Dataset class for the NER task"""

    def __init__(self, data_path:str=None, sentences:List[List[str]]=None, words_vocab:Vocab=None, use_pos:bool=False):
        """constructor of this class
        Args:
            data_path (str, optional): path where to load the whole Dataset, if passed it will have priority. Defaults to None.
            sentences (List[List[str]], optional): if Dataset is already loaded assume is a test set, pass sentences here. Defaults to None.
            words_vocab (Vocab, optional): if Dataset is already loaded assume is a test set,
            so you already have a vocab to pass here, in order to index the test set. Defaults to None.
            use_pos (bool, optional): whether to generate the pos tags or not. Defaults to False.
        """
        # since I'm not interested in going back from index to pos tag, and the upos tagset is fixed, use a plain dictionary and not a Vocab object
        upos2i = { "ADJ":1, "ADP":2, "ADV":3, "AUX":4, "CCONJ":5, "DET":6, "INTJ":7, "NOUN":8, "NUM":9, "PART":10,
                "PRON":11, "PROPN":12, "PUNCT":13, "SCONJ":14, "SYM":15, "VERB":16, "X":17} # leave 0-th index for padding

        # self.sentences = list of tokenized sentences, self.labels = labels for each token in self.sentences
        self.sentences, self.labels = sentences, None
        if data_path: # if data path is passed, override
            self.sentences, self.labels = self.parse_data(data_path)

        self.encoded_pos_tags = None
        if use_pos:
            # if using pos tags, initialize a new list of lists, containing the index-encoded pos tags matching indexes of words in self.sentences, e.g.:
            # [ ['he', 'was'], ['the', 'chief'] ] has tags [ ['PRON', 'AUX'], ['DET', 'NOUN'] ] represented as [ [11, 4], [6, 8] ]
            stanza.download(lang='en', processors='tokenize,pos', verbose=False) # note that due to this, the test container is slower: it needs to download these stanza english models each time
            pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True, verbose=False)
            doc = pos_tagger(self.sentences)
            self.encoded_pos_tags = [torch.LongTensor([upos2i[word.upos] for word in sent.words]) for sent in doc.sentences]
            assert len(self.encoded_pos_tags) == len(self.sentences)

        # index-encoded versions of previous lists, those contain tensors. TO BE FILLED VIA index_dataset(...) if it's not a train set
        self.encoded_sentences, self.encoded_labels = None, None
        if words_vocab: # if vocab is passed, index only the sentences directly
            self.encoded_sentences = []
            for i in range(len(self.sentences)): 
                sentence = self.sentences[i] # for each sentence
                self.encoded_sentences.append(torch.LongTensor([words_vocab[token.lower()] for token in sentence])) # encode it and put in object's field


    def parse_data(self, path: str) -> Tuple[List[List[str]], List[List[str]]]: # NOTE: this function is taken from evaluate.py
        """Parses the dataset from the given path 
        Args:
            path (str): Path of file containing data
        Returns:
            Tuple[List[List[str]], List[List[str]]]: First element is list of tokenized sentences,
            second element is list of labels for each sentence
        """
        all_sentences = []
        all_labels = []

        sentence = []
        labels = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#\tid"): # if we are starting a new sentence initialize lists
                    sentence = []
                    labels = []

                elif line == "": # if we ended a sentence then append lists of tokens
                    all_sentences.append(sentence)
                    all_labels.append(labels)

                else: # if we are inside a sentence keep appending tokens
                    token, label = line.split("\t")
                    sentence.append(token)
                    labels.append(label)

        assert len(all_sentences) == len(all_labels)
        return all_sentences, all_labels

    def index_dataset(self, words_vocabulary:Vocab, label_vocabulary:Vocab):
        """Indexes each token in the list of sentences with the index of words vocabulary.
            Indexes each token in the list of labels with the index of labels vocabulary.
        Args:
            words_vocabulary (Vocab): vocabulary of words
            label_vocabulary (Vocab): vocabulary of labels
        """
        self.encoded_sentences = []
        self.encoded_labels = []
        for i in range(len(self.sentences)): 
            sentence, labels = self.sentences[i], self.labels[i] # for each sentence-labels pair

            self.encoded_sentences.append(torch.LongTensor([words_vocabulary[token] for token in sentence])) # encode them and put in object's fields
            self.encoded_labels.append(torch.LongTensor([label_vocabulary[lab] for lab in labels]))

        assert len(self.encoded_sentences) == len(self.sentences)
        return

    def __len__(self) -> int:
        if self.encoded_sentences is None:
            raise RuntimeError("Trying to retrieve length but index_dataset has not been invoked yet!")
        return len(self.encoded_sentences)

    def __getitem__(self, idx:int) -> Dict[str,torch.LongTensor]:
        """returns a dict with idx-th encoded sentence, its pos tags and its list of labels
        Args:
            idx (int): index of sentence to retrieve
        Returns:
            Dict[str,torch.LongTensor]: a dictionary mapping "x":sentence, "pos": pos tags and "y":labels
        """
        if self.encoded_sentences is None:
            raise RuntimeError("Trying to retrieve elements but index_dataset has not been invoked yet!")
        elif self.encoded_labels is None: # then it's a test set, since I don't have ground truths but I have the sentences
            return {"x": self.encoded_sentences[idx]} if self.encoded_pos_tags is None else {"x": self.encoded_sentences[idx],
                                                                                            "pos": self.encoded_pos_tags[idx]}

        return {"x": self.encoded_sentences[idx],
                "y": self.encoded_labels[idx]} if self.encoded_pos_tags is None else {"x": self.encoded_sentences[idx],
                                                                                    "pos": self.encoded_pos_tags[idx],
                                                                                    "y": self.encoded_labels[idx]}

