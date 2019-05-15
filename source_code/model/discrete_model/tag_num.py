from discrete_model.comp_model import Crf, Bigru, BigruCrf, Linear
from discrete_model.shared_model import SentGru
from discrete_model.dataset_loader import batchload, MyTabularDataset
from discrete_model.cal_maxlen import sentence_maxlen_per_dialogue, sent_loader, sentence_maxlen_per_batch
from discrete_model.padding import all_preprocess
from discrete_model.utils import cal_accuracy, make_mask, loss_filtering, coder_mask
import numpy as np
import re
import json
import torch.utils.data
from torch.utils.data import DataLoader
import io,os
from torch import nn
from gensim.models import word2vec
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.autograd import Variable
from torchtext import data
import torch

def changeindex(inp):
    return inp[1:]

BATCH_SIZE = 128
HIDDEN_SIZE = 100
device = torch.device("cuda")

tag_to_ix = {'start_tag': 0, 'stop_tag': 29, 'pad_tag': 30}
tag_size = 31
working_path = '/home/jongsu/jupyter/pytorch_dialogue_ie/'
WV_PATH = '/home/jongsu/jupyter/pytorch_dialogue_ie/parameter/dialogue_wv'

wv_model = word2vec.Word2Vec(size=100, window=5, min_count=5, workers=4)
wv_model = word2vec.Word2Vec.load(WV_PATH)

my_fields = {'dial': ('Text', data.Field(sequential=True)),
             'emo': ('labels_1', data.Field(sequential=False)),
             'act': ('labels_2', data.Field(sequential=False))}
print("make data")
train_data = MyTabularDataset.splits(path=working_path, train='data_jsonfile/full_data_test.json', fields=my_fields)
train_data = sorted(train_data, key=lambda x: sentence_maxlen_per_dialogue(x))
train_data = train_data  # exclude dialogue which has extremely long sentence (0~11117 => 0~9999)
train = sorted(train_data, key=lambda x: -len(x.Text))  # reordering training dataset with number of sentences
# low index has much sentence because afterwards we use torch pad_sequence
print(train[0].labels_1)
print(train[0].labels_2)


def pad_cat_tag(emotion, act):
    i = 0
    new_tag = []
    while i < len(emotion):  # BATCH_SIZE
        emo, lenge = sent_loader(emotion[i][0])
        ac, lenga = sent_loader(act[i][0])
        j = 0
        inte = []  # append stop tag
        while j < len(emo):  # sent length
            inte.append(int(emo[j]))
            inte.append(int(ac[j])+10)
            j = j + 1
        inte.append(tag_to_ix['stop_tag'])  # append stop tag
        torch_inte = torch.tensor(inte)
        new_tag.append(torch_inte)  # str to int

        i = i + 1

    padded_tag = pad_sequence(new_tag, batch_first=True, padding_value=tag_to_ix['pad_tag'])

    # emotion+action string -> emotion+action numb + padding
    # batch*tag
    return padded_tag


def pad_cat_tag2(emotion, act):
    i = 0
    new_tag = []
    while i < len(emotion):  # BATCH_SIZE
        emo, lenge = sent_loader(emotion[i][0])
        ac, lenga = sent_loader(act[i][0])
        j = 0
        inte = []  # append stop tag
        while j < len(emo):  # sent length
            inte.append(int(emo[j])*4 + int(ac[j]))
            j = j + 1
        inte.append(tag_to_ix['stop_tag'])  # append stop tag
        torch_inte = torch.tensor(inte)
        new_tag.append(torch_inte)  # str to int

        i = i + 1

    padded_tag = pad_sequence(new_tag, batch_first=True, padding_value=tag_to_ix['pad_tag'])

    # emotion+action string -> emotion+action numb + padding
    # batch*tag
    return padded_tag


emotion_set = []
action_set = []
i = 0
while i < len(train):  # equals to BATCH_SIZE except last dataset
    emotion_set.append(train[i].labels_1)

    action_set.append(train[i].labels_2)

    i = i + 1

new_tag = pad_cat_tag(emotion_set, action_set)
new_tag2 = pad_cat_tag2(emotion_set, action_set)
print(new_tag)
print(new_tag.size())
emo = []
act = []
i = 0
new_tag = new_tag.numpy()
while i < len(new_tag):
    j = 0
    while j < len(new_tag[i]):
        if new_tag[i][j] < 10:
            emo.append(new_tag[i][j])
        elif new_tag[i][j] < 20:
            act.append(new_tag[i][j]-10)
        j = j + 1
    i = i + 1

all_dist = []
i = 0
while i < len(new_tag2):
    j = 0
    while j < len(new_tag2[i]):
        if new_tag2[i][j] < 29:
            all_dist.append(new_tag2[i][j])
        j = j + 1
    i = i + 1
print("counting")

i = 1
while i < 29:
    print("i = ", i)
    print(all_dist.count(i))
    i = i + 1

print(np.shape(emo))
print(np.shape(act))
print(emo)
print(act)
emo_0 = 0
emo_1 = 0
emo_2 = 0
emo_3 = 0
emo_4 = 0
emo_5 = 0
emo_6 = 0

act_0 = 0
act_1 = 0
act_2 = 0
act_3 = 0

i = 0
while i < len(emo):
    if emo[i] == 0:
        emo_0 = emo_0 + 1
    if emo[i] == 1:
        emo_1 = emo_1 + 1
    if emo[i] == 2:
        emo_2 = emo_2 + 1
    if emo[i] == 3:
        emo_3 = emo_3 + 1
    if emo[i] == 4:
        emo_4 = emo_4 + 1
    if emo[i] == 5:
        emo_5 = emo_5 + 1
    if emo[i] == 6:
        emo_6 = emo_6 + 1
    i = i + 1
i = 0
while i < len(act):
    if act[i] == 1:
        act_0 = act_0 + 1
    if act[i] == 2:
        act_1 = act_1 + 1
    if act[i] == 3:
        act_2 = act_2 + 1
    if act[i] == 4:
        act_3 = act_3 + 1
    i = i + 1

print(emo_0)
print(emo_1)
print(emo_2)
print(emo_3)
print(emo_4)
print(emo_5)
print(emo_6)
print("asdfasdf")
print(act_0)
print(act_1)
print(act_2)
print(act_3)