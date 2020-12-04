Chinese text classification
===========================
<br>This is the group project of MSBD5012 Team 26. The team members are: LIANG Minghui, LUO xinrui, SHEN Dinghui.:blush:<br>
In this project, we try to predict the class of Chinese news headlines using deep learning models, including TextCNN, FastText and BERT. We compare three different models and evaluate them using accuracies. 

Dataset:
--------
* The dataset used in this project is obtained by THUCNews (http://thuctc.thunlp.org/) related to Chinese news headlines and by Sogou News(https://github.com/Embedding/Chinese-Word -Vectors) respectively from github. <br>
* THUCNews is generated by filtering the historical data of The RSS feed of Sina News from 2005 to 2011.


Environment:
-----------
python 3.7 <br>
pytorch 1.1 <br>
tqdm <br>
sklearn <br>
tensorboardX <br>
Machine：three TiTAN GPUS

Operating insructions:
----------------------
##### TextCNN and FastText <br>
`cd Text-Classification-TextCNN-FastText`<br>
<br>
 #train and test on TextCNN <br> 
`python run.py --model TextCNN`

 #train and test on FastText, embedding is randomly initialized <br> 
`python run.py --model FastText --embedding random `

##### BERT <br>
###### Download of pre-trained model <br>
The bert model is placed in the bert_pretain directory. There are three files in the directory:
* pytorch_model.bin
* bert_config.json
* vocab.txt

download link:<br>
https://github.com/huggingface/transformers, <br>
bert_Chinese model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz <br>
vocab:https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt <br>
After decompression, put it in the corresponding directory as mentioned above and confirm the file name is correct.

`cd Text-Classification-BERT` <br>
<br>
#train and test on BERT
<br>`python run.py --model bert`

#train and test on BERT+CNN
<br> `python run.py --model bert_CNN `

Experimental results<br>
------------------------
We trained several models to predict the category of Chinese news headlines,BERT performs the highest accuracy in our text classification task, training time consumption of BERT is also the largest in these three models

| Model  | Accuracy |
| ------------- | ------------- |
| TextCNN   | 90.75%  |
| FastText  | 92.19%|
| BERT      | 94.69%|

Code composition <br>
-----------------
Data preprocessing：utils <br>
train and test：train_eval <br>
models：models.TextCNN, FastText, bert, bert_CNN <br>
main function：run <br>
