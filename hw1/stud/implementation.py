import torch
import numpy as np
from model import Model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from stud.myvocab import Vocab
from stud.mydataset import NERDataset
from stud.mymodel import ModHParams, NERModel
from typing import List, Dict

def build_model(device:str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates

    MODEL_FOLDER = "./model/"
    words_vocab = Vocab.load(MODEL_FOLDER+"words_vocab.json")
    labels_vocab = Vocab.load(MODEL_FOLDER+"labels_vocab.json")
    return StudentModel(MODEL_FOLDER+"best_weights.pt", words_vocab, labels_vocab, torch.device(device))

def prepare_test_batch(batch:List[Dict]) -> Dict[str,torch.Tensor]:
    """collate_fn for the test DataLoader, applies padding to data
    Returns:
        Dict[str,torch.Tensor]: a batch into a dictionary {x:data, pos:pos tags}
    """
    # extract features and labels from batch
    x = [sample["x"] for sample in batch]
    pos = None
    if "pos" in batch[0]: # if using pos tags
        pos = [sample["pos"] for sample in batch]

    # convert features to tensor and pad them
    x = pad_sequence(
            [torch.as_tensor(sample) for sample in x],
            batch_first=True
            )
    # eventually also pos tags
    if pos is not None:
        pos = pad_sequence(
            [torch.as_tensor(sample) for sample in pos],
            batch_first=True
            )

    return {"x": x, "pos": pos}


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, weights:str, words_vocab:Vocab, labels_vocab:Vocab, device):
        super().__init__()
        
        # create model hyperparameters, the layer weights will be overwritten with the trained one below
        # note that the default values of ModHParams constructor are the best-founded hyperparameters
        params = ModHParams(words_vocab, labels_vocab)

        self.model = NERModel(params) # initialize random-weighted model
        self.model.load_state_dict(torch.load(weights, map_location=device)) # load trained weights, on right device
        self.model.to(device)
        self.device = device
        self.words_vocab = words_vocab
        self.labels_vocab = labels_vocab


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        test_data = NERDataset(sentences=tokens, words_vocab=self.words_vocab, use_pos=True) # will be automatically indexed

        test_dataloader = DataLoader(test_data,
                                    batch_size=32,
                                    collate_fn=prepare_test_batch,
                                    shuffle=False)

        list_predictions = []
        self.model.eval()
        for batch in test_dataloader:
            x_data, pos_data = batch["x"].to(self.device), batch['pos']
            if pos_data is not None: pos_data = pos_data.to(self.device) # if using pos, move to device

            logits = self.model(x_data, pos_data)

            if self.model.crf is not None: # if using crf, then use its predictions
                mask = (x_data != 0)
                pred_indices = self.model.crf.viterbi_decode(logits, mask)
                list_predictions.extend([[self.labels_vocab.i2w[index] for index in sentence] for sentence in pred_indices]) # convert back from indices to labels

            else: # otherwise, compute argmax from logits
                torch_predictions = torch.argmax(logits, -1) # predictions in index format

                for i in range(x_data.shape[0]):
                    pred, x = torch_predictions[i], x_data[i]
                    mask = (x != 0) # to discard the padding
                    list_predictions.append([self.labels_vocab.i2w[index] for index in pred[mask].tolist()]) # convert back from indices to labels

        return list_predictions
