# let's start with all the imports
import json
from collections import Counter
from typing import Tuple, List


class Vocab():
    """Class that tries to mimic the behaviour of a torchtext.Vocab object, without using torchtext"""

    def __init__(self, counter:Counter, unk:bool=False, min_freq:int=1):
        """
        Args:
            counter (Counter): dictionary containing tokens:occurrencies pairs of our corpus
            unk (bool, optional): whether to set the <unk> token or not. If true, it means that
            we are creating the Vocab for the words, otherwise assume we are creating the one for labels. Defaults to False.
            min_freq (int, optional): tokens below this frequency will be omitted from the vocabulary. Defaults to 1.
        """
        self.w2i = {}
        self.i2w = {}
        index = 0
        if unk: # if we are creating words vocab, leave the 0-th index for padding and set the 1st index for <unk> token 
            self.w2i["<unk>"] = 1
            self.i2w[1] = "<unk>"
            index += 2
        
        for k,v in counter.items():
            if v >= min_freq: # keep only tokens above min_freq threshold
                self.w2i[k] = index
                self.i2w[index] = k
                index += 1
    
    def __len__(self) -> int:
        return len(self.w2i)

    def __getitem__(self, word:str) -> int:
        """return the index of a token in the w2i dictionary
        Args:
            word (str): the word of which we want the index
        Returns:
            int: the index of the word if present, else the index of <unk> token
        """
        return self.w2i[word] if word in self.w2i else self.w2i["<unk>"]
    
    def dump(self, path:str):
        """writes the object to file
        Args:
            path (str): path where to write the JSON serialized object
        """
        with open(path, "w") as outfile:
            to_write = {"w2i":self.w2i, "i2w":self.i2w}
            json.dump(to_write, outfile)
        return
    
    @staticmethod
    def load(path:str) -> "Vocab":
        """loads a Vocab object from file
        Args:
            path (str): path where to read the serialized object
        Returns:
            Vocab: the vocabulary object
        """
        with open(path, "r") as file:
            dictionary = json.load(file)
            w2i = dictionary["w2i"]
            i2w = dictionary["i2w"]
            voc = Vocab(Counter()) # create empty Vocab
            voc.w2i = w2i
            for k, v in i2w.items(): # since JSON converts int keys to str, revert this behaviour
                voc.i2w[int(k)] = v
        return voc

    @staticmethod
    def build_vocabs(sentences:List[List[str]], labels:List[List[str]], min_freq:int=1) -> Tuple["Vocab", "Vocab"]:
        """creates the vocabulary for the tokens and the one for the labels
        Args:
            sentences (List[List[str]]): list of list of tokens
            labels (List[List[str]]): list of list of labels, ordered in the same indexes wrt sentences parameter
            min_freq (int, optional): word tokens below this minimum frequency will be ignored.. Defaults to 1.
        Returns:
            Tuple[Vocab, Vocab]: the vocabulary for the tokens first, the one for the labels second
        """
        words_counter = Counter()
        labels_counter = Counter()

        for i in range(len(sentences)): # for each sentence
            tokens = sentences[i]
            lab = labels[i]
            assert len(tokens) == len(lab)

            for j in range(len(tokens)): # for each token, update the counters
                words_counter[tokens[j]] += 1
                labels_counter[lab[j]] += 1

        return Vocab(words_counter, unk=True, min_freq=min_freq), Vocab(labels_counter)