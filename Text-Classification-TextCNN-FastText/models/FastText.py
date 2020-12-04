# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        self.train_path = dataset + '/data/train.txt'                                # train data
        self.dev_path = dataset + '/data/dev.txt'                                    # dev data
        self.test_path = dataset + '/data/test.txt'                                  # test data
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # class category
        self.vocab_path = dataset + '/data/vocab.pkl'                                # vocab
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # model training results
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained word vector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # device

        self.dropout = 0.5                                              # drop-out
        self.require_improvement = 1000                                 # if the effect of more than 1000 batches has not improved,end the training
        self.num_classes = len(self.class_list)                         # the number of classes
        self.n_vocab = 0                                                # the size of vocab
        self.num_epochs = 20                                            # the number of epoch
        self.batch_size = 128                                           # the size of mini-batch
        self.pad_size = 32                                              # The length of each sentence
        self.learning_rate = 1e-3                                       # learning rate
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # Word vector dimension
        self.hidden_size = 256                                          # Hidden layer siz
        self.n_gram_vocab = 250499                                      # the size of ngram vocab


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
