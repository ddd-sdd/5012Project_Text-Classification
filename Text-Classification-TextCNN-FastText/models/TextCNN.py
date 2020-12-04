# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # training dataset
        self.dev_path = dataset + '/data/dev.txt'                                    # dev
        self.test_path = dataset + '/data/test.txt'                                  # test
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # the class category
        self.vocab_path = dataset + '/data/vocab.pkl'                                # vocab
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # model training results
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained word vector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # device

        self.dropout = 0.5                                              # drop out
        self.require_improvement = 1000                                 # if the effect of more than 1000 batches has not improved,end the training
        self.num_classes = len(self.class_list)                         # the number of classes
        self.n_vocab = 0                                                # the size of vocabs
        self.num_epochs = 20                                            # the number of epoch
        self.batch_size = 128                                           # the size of mini-batch
        self.pad_size = 32                                              # The length of each sentence
        self.learning_rate = 1e-3                                       # learning rate
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # Word vector dimension
        self.filter_sizes = (2, 3, 4)                                   # the fize of filter
        self.num_filters = 256                                          # the number of filtter


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
