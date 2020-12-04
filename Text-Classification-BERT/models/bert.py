# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # train
        self.dev_path = dataset + '/data/dev.txt'                                    # dev
        self.test_path = dataset + '/data/test.txt'                                  # test
        self.log_path = dataset + '/log/' + self.model_name
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # class of category
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # model training results
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


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # the input sentence
        mask = x[2]  # masked the padding,eg：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
