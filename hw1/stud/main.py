# this script was executed to train the models, let's start with all the imports
import os
import torch
import random
import gensim.downloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from mydataset import NERDataset
from myvocab import Vocab
from mytrainer import Trainer
from mymodel import ModHParams, NERModel
from myutils import prepare_batch, load_pretrained_embeddings

# if true, execute the grid search on hyperparameters, else just train the manually-set model at the end of this script
GRID_SEARCH = False

# fix the seed to allow reproducibility
SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#data paths, please note that this script ran on my local computer
MODEL_FOLDER = "../../model/"
DATA_FOLDER = "../../data/"
TRAIN_DATA = DATA_FOLDER + "train.tsv"
VAL_DATA = DATA_FOLDER + "dev.tsv"

#set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate datasets and vocabs
train_data = NERDataset(TRAIN_DATA, use_pos=True)
words_vocab, labels_vocab = Vocab.build_vocabs(train_data.sentences, train_data.labels) # instantiate the vocabs from train set
words_vocab.dump(MODEL_FOLDER+"words_vocab.json") # save them
labels_vocab.dump(MODEL_FOLDER+"labels_vocab.json")
train_data.index_dataset(words_vocab, labels_vocab) # encodes the dataset with the vocabulary
val_data = NERDataset(VAL_DATA, use_pos=True)
val_data.index_dataset(words_vocab, labels_vocab)

# create DataLoaders
workers = min(os.cpu_count(), 4)
train_dataloader = DataLoader(train_data,
                            batch_size=32,
                            collate_fn=prepare_batch,
                            num_workers=workers,
                            shuffle=True)

valid_dataloader = DataLoader(val_data,
                            batch_size=32,
                            collate_fn=prepare_batch,
                            num_workers=workers,
                            shuffle=False)

if GRID_SEARCH:
    # grid search on some hyperparameters, I modified these every now and then to search faster according to some previous results
    # you can see all the hyperparameters I tried (in different combinations) in the comments beside of the for loops
    for glove in [300]: #50, 100, 200, 300
        # pretrained = gensim.downloader.load("word2vec-google-news-300")
        pretrained = gensim.downloader.load("glove-wiki-gigaword-"+str(glove))

        for pos_dim in [25]: # 10, 25, 50, 100
            for hidden in [256]: # 128, 256
                for lstm_layers in [3]: # 2, 3
                    for lrate in [0.0005, 0.001, 0.005]: # 1e-4, 0.0005, 0.001, 0.005, 0.01
                        for wdecay in [0, 1e-6, 1e-5, 1e-4]: # 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
                            for fc in [1]: # 1, 2, 3, 4
                                
                                # create model hyperparameters
                                params = ModHParams(words_vocab,
                                                    labels_vocab,
                                                    word_embedding_dim=pretrained.vectors.shape[1],
                                                    word_embeddings=load_pretrained_embeddings(pretrained, words_vocab, True),
                                                    hidden_dim=hidden,
                                                    lstm_layers=lstm_layers,
                                                    pos_embedding_dim=pos_dim,
                                                    fc_layers=fc,
                                                    use_crf=True)

                                # instantiate and train the model
                                nermodel = NERModel(params)
                                trainer = Trainer(
                                    model=nermodel,
                                    loss_function=nn.CrossEntropyLoss(),
                                    optimizer=torch.optim.Adam(nermodel.parameters(), lr=lrate, weight_decay=wdecay), # weight decay = L2 regularization
                                    label_vocab=labels_vocab,
                                    device=device
                                    )

                                metrics = trainer.train(train_dataloader, valid_dataloader, 60, 9, MODEL_FOLDER+"dump.pt")
                                with open("./results/results6.txt", "a") as f:
                                    f.write('GLOVE {}, POS {}, HIDDEN {}, LAYERS {}, LR {}, DECAY {}, FC {}: BEST VALID F1: {:0.3f}% AT EPOCH {}, TRAINED FOR {} EPOCHS\n'.format(
                                                                                        glove, pos_dim, hidden, lstm_layers, lrate, wdecay, fc,
                                                                                        max(metrics["valid_f1_history"])*100,
                                                                                        np.argmax(metrics["valid_f1_history"])+1,
                                                                                        len(metrics["train_history"]))
                                                                                        )
                                Trainer.plot_logs(metrics, './images/loss'+'_GLOVE{}_POS{}_HIDDEN{}_LAYERS{}_LR{}_DECAY{}_FC{}'.format(glove,
                                                                                                                                pos_dim,
                                                                                                                                hidden,
                                                                                                                                lstm_layers,
                                                                                                                                lrate, wdecay,
                                                                                                                                fc)+'.png') # loss plot
else:
    # after this coarse-grained hyperparameter tuning, I can find the best candidate model at each step by looking in the resultsN.txt file,
    # where greater N indicates a file obtained during a subsequent phase of hyperparameter search (e.g. after adding pos tags).
    # After finding it, I re-train it to save the weights and plot the classification report and the confusion matrix 
    # as well the loss during training and validation. Note that results1.txt is not the absolute first batch of tests, because with the
    # initial architecture the model was just manually-tested each time.

    pretrained = gensim.downloader.load("glove-wiki-gigaword-300")

    # create model hyperparameters, note that the default values of ModHParams constructor are already the best-founded hyperparameters
    params = ModHParams(words_vocab,
                        labels_vocab,
                        word_embeddings=load_pretrained_embeddings(pretrained, words_vocab, True))

    # instantiate and train the model
    nermodel = NERModel(params)
    trainer = Trainer(
        model=nermodel,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(nermodel.parameters(), lr=0.001, weight_decay=1e-5), # weight decay = L2 regularization
        label_vocab=labels_vocab,
        device=device
        )

    metrics = trainer.train(train_dataloader, valid_dataloader, 60, 9, MODEL_FOLDER+"best_weights.pt")
    print( 'BEST VALID F1: {:0.4f}% AT EPOCH {}'.format(max(metrics["valid_f1_history"])*100, np.argmax(metrics["valid_f1_history"])+1) )
    trainer.generate_cm('./cm.png') # confusion matrix
    Trainer.plot_logs(metrics, './losses.png') # loss plot
