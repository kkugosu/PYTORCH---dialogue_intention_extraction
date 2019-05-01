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

def maxindex(arry):
    i = 0
    j = 0
    maxidx = 0
    while i < len(arry):
        if j < arry[i]:
            j = arry[i]
            maxidx = i
        i = i + 1
    return maxidx

def max_indexset_per_batch(ary):
    indexset = []
    i = 0
    while i < len(ary):
        small_indexset = []
        j = 0

        while j < len(ary[i]):
            small_indexset.append(maxindex(ary[i][j]))
            j = j + 1
        indexset.append(small_indexset)
        i = i + 1
    return indexset


def cal_accuracy(model_predict, real_tag):  # per dialogue
    '''
    Args:
        model_predict
            model predicted tags
        real_tag
            real tags
        tag_len
            tag len

    Yields:
        accuracy

    Example:

    real = torch.tensor([  0,   2,   1,   2,   1,   1,   1,   1,   1,   2,   1,   1, 1,   1,  29], device='cuda:0')
    model = [2, 1, 2, 1, 17, 1, 17, 1, 3, 1, 17, 1, 17, 29]
    taglen = len(model)

    (npreal[tagseq+1]//4) real emotion
    (model_predict[tagseq]//4) model emotion

    (npreal[tagseq+1]%4) real action
    (model_predict[tagseq]%4) model action

    emotion err = 0.3076923076923077
    action err = 0.07692307692307693
    accuracy = 0.8076923076923077
    '''
    tag_len = len(model_predict)
    npreal = real_tag.cpu().numpy()  #
    tagseq = 0
    emotiontag = []
    actiontag = []
    emoerr = 0
    acterr = 0
    while tagseq < (tag_len - 1):

        emotiontag = np.append(emotiontag, npreal[tagseq + 1] // 4)
        actiontag = np.append(actiontag, npreal[tagseq + 1] % 4)

        if (npreal[tagseq + 1] // 4) != (model_predict[tagseq] // 4):
            emoerr = emoerr + 1

        if (npreal[tagseq + 1] % 4) != (model_predict[tagseq] % 4):
            acterr = acterr + 1

        tagseq = tagseq + 1

    return 1 - (emoerr / tagseq + acterr / tagseq) / 2



BATCH_SIZE = 64
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

print("make data finish")

dataseq = torch.arange(end=len(train), dtype=torch.int)

# dataseq = torch.zeros((len(train)), dtype=torch.int).fill_(5950)

grucrf = BigruCrf(tag_to_ix, HIDDEN_SIZE).cuda()
gru = Bigru(tag_to_ix, HIDDEN_SIZE).cuda()
crf = Crf(tag_to_ix, HIDDEN_SIZE).cuda()
linear = Linear(tag_size, HIDDEN_SIZE).cuda()
# this convert batch*(dialogue_length*sent_vec(float)) -> batch*(dialogue_tag,score)

sent = SentGru(HIDDEN_SIZE, bidirectional=True, device=device).cuda()

print("now ready")


filtering_value = 3
crf.load_state_dict(torch.load(working_path + 'parameter/crf.pth'))
grucrf.load_state_dict(torch.load(working_path + 'parameter/crf_gru.pth'))
sent.load_state_dict(torch.load(working_path + 'parameter/shared.pth'))


sent.load_state_dict(torch.load(working_path + 'parameter/linear_share.pth'))
linear.load_state_dict(torch.load(working_path + 'parameter/linear.pth'))


iter_num = 0
k = 0
while iter_num < 1:
    iter_num = iter_num + 1
    batchnum = 1
    linear_acc = 0
    for batch_data in batchload(train, repeat=False, batchsize=BATCH_SIZE, data_seq=dataseq):
        # load txt data from jsonfile

        print('new_batch----------------')
        print("sent_maxlen = ", sentence_maxlen_per_batch(batch_data))
        print("dial_len_range = ", len(batch_data[0].Text), " - ", len(batch_data[BATCH_SIZE-1].Text))

        batchnum = batchnum + 1

        #    continue

        new_dial, new_tag, dial_leng = all_preprocess(sent, batch_data)
        # load batch* (dialogue_length*sent_vec(float)) -> new_dial
        # load batch* tag -> new_tag
        # load batch* dial_leng
        # print(np.shape(make_mask(dial_leng)))
        # print(np.shape(new_dial))
        # print(new_tag[0].cpu().numpy())
        # print(coder_mask(new_tag[0].cpu().numpy(), 31, True).view(-1, 31))

        tensormask = make_mask(dial_leng)

        my_output = linear(tensormask, new_dial)
        my_output_tag = max_indexset_per_batch(my_output)

        i = 0
        sum_acc = 0
        while i < BATCH_SIZE:
            sum_acc = sum_acc + cal_accuracy(my_output_tag[i], new_tag[i])

            i = i + 1
        print(sum_acc / BATCH_SIZE)

        linear_acc = linear_acc + sum_acc/BATCH_SIZE




        # loss = grucrf.neg_log_likelihood(make_mask(dial_leng), new_dial, new_tag, BATCH_SIZE)
        # loss = crf.neg_log_likelihood(make_mask(dial_leng), new_dial, new_tag, BATCH_SIZE)

        newary_ = []
        # loss, newary_ = loss_filtering(loss, filtering_value, newary_, k

        # torch.save(crf.state_dict(), working_path + 'parameter/crf.pth')
        # torch.save(linear.state_dict(), working_path + 'parameter/linear.pth')
        # torch.save(sent.state_dict(), working_path + 'parameter/linear_share.pth')
        '''
                if batchnum == 50:
            break

        unuselist = [new_dial, new_tag, dial_leng]
        del unuselist

        if k % 10 != 0:
            # torch.save(sent.state_dict(), working_path + 'parameter/shared.pth')
            # torch.save(grucrf.state_dict(), working_path + 'parameter/crf_gru.pth')  # 3.53 save with dummy
            dummy_input = [make_mask(dial_leng), new_dial]

            print("tag = ", new_tag[7])
            print("expect = ", grucrf(BATCH_SIZE, dummy_input, seq=7)[1])
            print("accuracy = ", cal_accuracy(grucrf(BATCH_SIZE, dummy_input, seq=7)[1], new_tag[7]))

            print(loss)
        if k == int(len(train_data) / BATCH_SIZE) * iter_num:
            break

        if k % int(len(train_data) / BATCH_SIZE) == 0:
            newary = newary_
            newary_ = []
        '''
    print("aaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(linear_acc / batchnum)
