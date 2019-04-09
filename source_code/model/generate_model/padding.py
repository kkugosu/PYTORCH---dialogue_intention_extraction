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







def makewv(_wv_model, target, batch_size):
    targetwv = numerize_sent(target[0], len(target[0]), _wv_model)
    batchnum = 0
    while batchnum < batch_size:
        if batchnum == 0:
            targetwv = target
        else:
            targetwv = torch.cat((targetwv, target), 0)
        # wv = numerize_sent(target[batchnum], len(target[batchnum]))
        # targetwv.append(wv)

        batchnum = batchnum + 1
    targetwv = torch.tensor(targetwv).type(torch.cuda.FloatTensor)
    return targetwv


def sent_loader(sentence):

    """
    pre_process per sentence

    :param sentence:
    :return:
    """

    result = []
    for elem in sentence.split(' '):
        if elem != '':
            result = np.append(result, elem)
    return result, len(result)


def numerize_sent(sent, len_sent, _wv_model): #input output to cuda
    i = 0
    n_sent = []
    while i < len_sent:
        if sent[i] == '<pad>':
            n_sent.append(np.zeros(100))

        elif sent[i] == '<stop_tag>':
            n_sent.append(np.ones(100))
        else:
            try:
                n_sent.append(_wv_model.wv[sent[i]])
            except:
                n_sent.append(np.zeros(100))

        i = i + 1
    return n_sent


def pad_dial(last_v):
    leng_set = []
    i = 0
    while i < len(last_v):  # BATCH_SIZE
        leng_set.append(len(last_v[i]))  # sentence num
        i = i + 1
    padded_dial = pad_sequence(last_v, batch_first=True)  # append padtag vector
    # print('max_dialogue_length',len(padded_dial[0]))

    return padded_dial, leng_set


def batch_numerical(sent_set, _wv_model):
    numeric_batch = []  # numerized batch
    i = 0
    while i < len(sent_set):  # BATCH_SIZE
        dial = []  # numerized dialogue
        j = 0
        while j < len(sent_set[i]):  # per dialogue
            '''
            sent_set[i][j] ['Is' 'this' 'your' 'new' 'teacher' '?' '<pad>']
            sent_set[i][j] ['Yes' ',' 'it' 'is' '.' '<pad>' '<pad>']
            sent_set[i][j] ['Is' 'she' 'short' '?' '<pad>' '<pad>' '<pad>']
            sent_set[i][j] ['No' ',' 'she' '’' 's' 'average' '.']
            sent_set[i][j] ['What' 'color' 'are' 'her' 'eyes' '?' '<pad>']
            sent_set[i][j] ['They' '’' 're' 'dark' 'gray' '.' '<pad>']
            sent_set[i][j] ['What' 'color' 'is' 'her' 'hair' '?' '<pad>']
            sent_set[i][j] ['It' '’' 's' 'blond' '.' '<pad>' '<pad>']
            sent_set[i][j] ['And' 'how' 'old' 'is' 'she' '?' '<pad>']
            sent_set[i][j] ['I' 'don' '’' 't' 'know' '.' '<pad>']
            sent_set[i][j] ['<stop_tag>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>']
            '''
            dial.append(numerize_sent(sent_set[i][j], len(sent_set[i][j]), _wv_model))  # numerized_sentence
            j = j + 1
        numeric_batch.append(dial)
        i = i + 1
    # batch * sent_num * sent_leng * wv -> numeric_batch
    return numeric_batch


def make_batch2sent(new):

    for_sentmodel = torch.tensor(new[0]).cuda()
    batchnum = 1
    while batchnum < len(new):  # BATCH_SIZE
        for_sentmodel = torch.cat((for_sentmodel, torch.tensor(new[batchnum]).cuda()))
        batchnum = batchnum + 1
    sentbatch_len = len(for_sentmodel)
    # batch * sent_num * sent_leng * wv -> all_sent_num * sent_leng * wv
    return sentbatch_len, for_sentmodel


def pad_tag(batch_data, tag_to_ix, _device):

    emotion_set = []
    action_set = []
    temp_tag = []
    de_tag = []

    i = 0
    while i < len(batch_data):  # equals to BATCH_SIZE except last dataset
        emotion_set.append(batch_data[i].labels_1)
        action_set.append(batch_data[i].labels_2)
        i = i + 1

    i = 0
    while i < len(emotion_set):  # BATCH_SIZE
        emo, lenge = sent_loader(emotion_set[i][0])
        ac, lenga = sent_loader(action_set[i][0])
        j = 0
        inte = []
        # inte.append(tag_to_ix['start_tag']) #append stop tag
        while j < len(emo) - 1:  # sent length
            inte.append(int(emo[j]) * 4 + int(ac[j]))
            j = j + 1
        de_tag.append(int(emo[j]) * 4 + int(ac[j]))  # <-for decoding sent
        # inte.append(tag_to_ix['stop_tag']) #append stop tag
        torch_inte = torch.tensor(inte)
        temp_tag.append(torch_inte)  # str to int
        i = i + 1

    en_tag = pad_sequence(temp_tag, batch_first=True, padding_value=tag_to_ix['pad_tag'])

    # emotion+action string -> emotion+action numb + padding
    # batch*tag
    # in new preprocess, remake pad cat tag to split last sent
    '''
    en_tag representation 0 = start_tag, 29 = stop_tag, 30 = pad_tag

    tensor([   2,   1,   2,   1,   2,   1,   2,   1,   2,   1]) except1
    tensor([   2,   1,   2,   1,   2,   1,   2,   1,   2,   1]) except1
    tensor([  21,   1,   1,   1,  30,  30,  30,  30,  30,  30]) except1
    tensor([   2,   1,   2,   1,  30,  30,  30,  30,  30,  30]) except1
    tensor([   1,   1,   1,   1,  30,  30,  30,  30,  30,  30]) except1
    tensor([   2,   1,   2,   1,  30,  30,  30,  30,  30,  30]) except1
    tensor([  17,  17,  17,  30,  30,  30,  30,  30,  30,  30]) except17
    tensor([   1,   1,   1,  30,  30,  30,  30,  30,  30,  30]) ..
    tensor([   3,   2,  30,  30,  30,  30,  30,  30,  30,  30]) ..
    tensor([  17,  17,  30,  30,  30,  30,  30,  30,  30,  30]) ..
    tensor([   2,   2,  30,  30,  30,  30,  30,  30,  30,  30]) ..

    de_tag representation
    [1, 1, 1, 1, 1, 1, 17, 1, 2, 17, 2...] len = batchsize

    '''
    return en_tag, de_tag


def pad_text(sent_m, _wv_model, batch_data, _device):

    en_text_1 = []
    de_text = []
    all_seq_len = []
    de_len = []
    sentnum_per_batch = []
    batch_size = len(batch_data)
    i = 0
    maxleng = 1
    while i < batch_size:  # almost equal to BATCH_SIZE
        j = 0
        temp = []
        sentnum_per_batch.append(len(batch_data[i].Text) - 2)  # -stoptag-lastsent
        # save sentnum per dialogue
        while j < len(batch_data[i].Text) - 2:  # -lastsent
            sent, leng = sent_loader(batch_data[i].Text[j])
            # convert text to word_list, word_list_length
            all_seq_len.append(leng)
            if leng > maxleng:
                maxleng = leng
            temp.append(sent)
            j = j + 1
        sent, leng = sent_loader(batch_data[i].Text[j])
        de_text.append(sent)
        de_len.append(leng)
        # temp.append(["<stop_tag>"])
        # all_seq_len.append(1) #stoptag
        en_text_1.append(temp)
        i = i + 1
    i = 0

    while i < batch_size:  # almost equal to BATCH_SIZE
        j = 0
        while j < len(en_text_1[i]):
            while len(en_text_1[i][j]) < maxleng:
                en_text_1[i][j] = np.append(en_text_1[i][j], "<pad>")
            j = j + 1
        i = i + 1
    i = 0

    while i < batch_size:  # almost equal to BATCH_SIZE
        while len(de_text[i]) < maxleng:
            de_text[i] = np.append(de_text[i], "<pad>")
        i = i + 1

    # batch * sentnum * (word_list+pad) -> en_text_1
    # batch * sentnum * (word_list_length) -> all_seq_len
    # batch * (sentnum) -> sentnum_per_batch
    # batch * (word_list + pad) -> dn
    # batch * leng -> dl
    # in new preprocess, remake pad batch to split last sent

    '''
    print(dn[0])
    print(dn[1])
    print(dn[2])
    print(dl)

    ['Thank' 'you' 'so' 'much' '.' 'You' 'guys' 'are' 'really' 'responsible'
     '.' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>']
    ['Alright' ',' 'please' 'show' 'me' 'what' 'you' 'have' '.' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>']
    ['Bye' '!' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>' '<pad>'
     '<pad>']
    [11, 9, 2, 3, 12, 15, 2, 5, 20, 24, 6, 12, 10, 4, 4, 11, 6, 12, 7, 6, 15, 
    16, 12, 2, 17, 4, 7, 2, 2, 12, 20, 9, 2, 6, 9, 4, 21, 3, 7, 5, 4, 18, 7, 
    7, 10, 19, 7, 31, 9, 7, 7, 4, 12, 4, 4, 15, 4, 12, 9, 6, 2, 2, 6, 10, 16, 
    4, 6, 7, 5, 22, 9, 11, 11, 10, 4, 4, 2, 2, 4, 17, 15, 5, 16, 14, 5, 3, 22, 
    2, 2, 20, 7, 2, 2, 3, 2, 3, 6, 6, 5, 5]
    '''

    '''
    en_text_1[0]
    [array(['Is', 'this', 'your', 'new', 'teacher', '?', '<pad>'], dtype='<U32'), 
      array(['Yes', ',', 'it', 'is', '.', '<pad>', '<pad>'], dtype='<U32'), 
      array(['Is', 'she', 'short', '?', '<pad>', '<pad>', '<pad>'], dtype='<U32'), 
      array(['No', ',', 'she', '’', 's', 'average', '.'], dtype='<U32'), 
      array(['What', 'color', 'are', 'her', 'eyes', '?', '<pad>'], dtype='<U32'), 
      array(['They', '’', 're', 'dark', 'gray', '.', '<pad>'], dtype='<U32'), 
      array(['What', 'color', 'is', 'her', 'hair', '?', '<pad>'], dtype='<U32'),
      array(['It', '’', 's', 'blond', '.', '<pad>', '<pad>'], dtype='<U32'), 
      array(['And', 'how', 'old', 'is', 'she', '?', '<pad>'], dtype='<U32'), 
      array(['I', 'don', '’', 't', 'know', '.', '<pad>'], dtype='<U32'), 
      array(['<stop_tag>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'], dtype='<U32'), 

    all_seq_len[0]
    [6, 5, 4, 7, 6, 6, 6, 5, 6, 6, 1, 
    6, 5, 4, 7, 6, 6, 6, 5, 6, 6, 1, 
    6, 5, 7, 2, 1, 
    5, 6, 5, 6, 1, 
    4, 4, 6, 4, 1, 
    6, 5, 5, 5, 1, 
    4, 4, 3, 1, 
    5, 5, 4, 1, 
    3, 3, 1, 
    3, 2, 1, 
    3, 2, 1, 
    3, 3, 1, 
    4, 2, 1, 
    4, 2, 1, 
    3, 3, 1, 
    4, 3, 1, 
    4, 3, 1, 
    4, 3, 1, 
    4, 3, 1, 
    3, 3, 1, 
    4, 2, 1, 
    3, 3, 1, 
    5, 4, 1, 
    4, 3, 1, 
    .
    .
    .
    sentnum_per_batch
    [11, 11, 5, 5, 5, 5, 4, 4, 3, 3, 3, 3, .....]

    '''
    en_text_2 = batch_numerical(en_text_1, _wv_model)
    # batch * sent_num * sent_leng * wv -> en_text_2
    sentbatch_len, for_sentmodel = make_batch2sent(en_text_2)
    # batch * sent_num * sent_leng * wv -> all_sent_num(new_batch) * sent_leng * wv
    # for_sentmodel2 -> torch.Size([326, 7, 100])
    # sentbatch_len -> 326
    # hidden_state = torch.tensor(np.zeros((2, sentbatch_len, 100)), dtype=torch.float, device=device,
    #                            requires_grad=False)
    # for_sentmodel2 = torch.tensor(np.transpose(for_sentmodel, [1, 0, 2]), dtype=torch.float, device=device,
    #                              requires_grad=False)

    # for_sentmodel2 -> torch.Size([7, 326, 100])
    # pre_crf_gru = sent_m(for_sentmodel2, hidden_state, all_seq_len)

    pre_crf_gru = sent_m(for_sentmodel, sentbatch_len, all_seq_len)
    '''
    print(pre_crf_gru[0])
    print(pre_crf_gru[1])
    print(pre_crf_gru[2])
    print(pre_crf_gru[3])
    print(pre_crf_gru[4])
    print(pre_crf_gru[5])
    print(pre_crf_gru[6])
    print(pre_crf_gru[7])
    print(pre_crf_gru[8])
    print(pre_crf_gru[9])
    '''
    # pre_crf_gru -> torch.Size([326, 100])
    # ----------------------------------------------------- sent network
    # all_sent_num(new_batch) * sent_leng * wv -> all_sent_num(new_batch) * wv
    last_var = torch.split(pre_crf_gru, sentnum_per_batch)
    # all_sent_num(new_batch) * wv -> batch * sent_num * wv
    '''
    last_var

    [11,100]
    [11,100]
    [5,100]
    [5,100]
    [5,100]
    .
    .
    '''
    en_text, en_len = pad_dial(last_var)
    # batch * sent_num * wv -> batch * (sent_num + pad) * wv
    # save dial_leng for masking
    '''
    en_text

    [11,100]
    [11,100]
    [11,100]
    [11,100]
    [11,100]
    .
    .
    '''
    targetwv = makewv(_wv_model, de_text, batch_size)

    return en_text, en_len, targetwv, de_len




