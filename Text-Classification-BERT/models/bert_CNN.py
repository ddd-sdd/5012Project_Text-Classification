# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # train
        self.dev_path = dataset + '/data/dev.txt'                                    # dev
        self.test_path = dataset + '/data/test.txt'                                  # test
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # the class of category
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # model train results
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # device

        self.require_improvement = 1000                                 # end training if it is not improved in 1000 epochs
        self.num_classes = len(self.class_list)                         # the number of categories
        self.num_epochs = 3                                             # the number of epoch
        self.batch_size = 128                                           # the size of mini-batch
        self.pad_size = 32                                              # the length of every sentence
        self.learning_rate = 5e-5                                       # learning rate
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # the size of filter
        self.num_filters = 256                                          # the number of channels
        self.dropout = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # the input sentence
        mask = x[2]  # Mask the padding part, and a size of the sentence, the padding part is represented by 0，eg：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out
