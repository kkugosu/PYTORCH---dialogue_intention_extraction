import numpy as np
import torch
from gensim.models import word2vec
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

WV_PATH = '/home/jongsu/jupyter/pytorch_dialogue_ie/parameter/dialogue_wv'

wv_model = word2vec.Word2Vec(size=100, window=5, min_count=5, workers=4)
wv_model = word2vec.Word2Vec.load(WV_PATH)
tag_to_ix = {'start_tag': 0, 'stop_tag': 29, 'pad_tag': 30}


def pad_dial(last_v):
    leng_set = []
    i = 0
    while i < len(last_v):  # BATCH_SIZE
        leng_set.append(len(last_v[i]))  # sentence num
        i = i + 1
    padded_dial = pad_sequence(last_v, batch_first=True)  # append padtag vector
    # print('max_dialogue_length',len(padded_dial[0]))

    return padded_dial, leng_set


def sent_loader(sentence):  # pre_process per sentence
    result = []
    for elem in sentence.split(' '):
        if elem != '':
            result = np.append(result, elem)
    return result, len(result)


def make_batch2sent(new):
    for_sentmodel = []
    batchnum = 0
    while batchnum < len(new):  # BATCH_SIZE
        for_sentmodel = for_sentmodel + new[batchnum]
        batchnum = batchnum + 1
    sentbatch_len = len(for_sentmodel)
    # batch * sent_num * sent_leng * wv -> all_sent_num * sent_leng * wv
    for_sentmodel = torch.FloatTensor(for_sentmodel).cuda()
    return sentbatch_len, for_sentmodel


def pad_cat_tag(emotion, act):
    i = 0
    new_tag = []
    while i < len(emotion):  # BATCH_SIZE
        emo, lenge = sent_loader(emotion[i][0])
        ac, lenga = sent_loader(act[i][0])
        j = 0
        inte = [tag_to_ix['start_tag']]  # append stop tag
        while j < len(emo):  # sent length
            inte.append(int(emo[j]) * 4 + int(ac[j]))
            j = j + 1
        inte.append(tag_to_ix['stop_tag'])  # append stop tag
        torch_inte = torch.tensor(inte)
        new_tag.append(torch_inte)  # str to int

        i = i + 1

    padded_tag = pad_sequence(new_tag, batch_first=True, padding_value=tag_to_ix['pad_tag'])

    # emotion+action string -> emotion+action numb + padding
    # batch*tag
    return padded_tag


def numerize_sent(sent, len_sent):
    i = 0
    n_sent = []
    while i < len_sent:
        if sent[i] == '<pad>':
            n_sent.append(np.zeros(100))

        elif sent[i] == '<stop_tag>':
            n_sent.append(np.ones(100))
        else:
            try:
                n_sent.append(wv_model.wv[sent[i]])
            except:
                n_sent.append(np.zeros(100))

        i = i + 1
    return n_sent


def batch_numerical(sent_set):
    numeric_batch = []  # numerized batch
    i = 0
    while i < len(sent_set): #BATCH_SIZE
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
            dial.append(numerize_sent(sent_set[i][j], len(sent_set[i][j])))  # numerized_sentence
            j = j + 1
        numeric_batch.append(dial)
        i = i + 1
    # batch * sent_num * sent_leng * wv -> numeric_batch
    return numeric_batch


def pad_batch(minibatch):
    i = 0
    new_batch = []
    leng_set = []
    maxleng = 1
    sentnum_per_dialogue = []
    while i < len(minibatch):  # almost equal to BATCH_SIZE
        j = 0
        temp = []
        sentnum_per_dialogue.append(len(minibatch[i].Text))  # '' = stoptag
        # save sentnum per dialogue
        while j < len(minibatch[i].Text) - 1:
            sent, leng = sent_loader(minibatch[i].Text[j])

            # convert text to word_list, word_list_length
            leng_set.append(leng)
            if leng > maxleng:
                maxleng = leng

            temp.append(sent)
            j = j + 1

        temp.append(["<stop_tag>"])
        leng_set.append(1)  # stoptag
        new_batch.append(temp)

        i = i + 1

    i = 0
    while i < len(minibatch):  # almost equal to BATCH_SIZE
        j = 0
        while j < len(new_batch[i]):

            while len(new_batch[i][j]) < maxleng:
                new_batch[i][j] = np.append(new_batch[i][j], "<pad>")

            j = j + 1
        i = i + 1
    # batch * sentnum * (word_list+pad) -> new_batch
    # batch * sentnum * (word_list_length) -> leng_set
    # batch * (sentnum) -> sentnum_per_dialogue

    return new_batch, leng_set, sentnum_per_dialogue


def all_preprocess(sent, batch_data):
    # sorted with dialogue length
    # print(batch_data[0].Text)
    batch_size = len(batch_data)
    #######################################################
    emotion_set = []
    action_set = []
    i = 0
    while i < len(batch_data):  # equals to BATCH_SIZE except last dataset
        emotion_set.append(batch_data[i].labels_1)

        action_set.append(batch_data[i].labels_2)

        i = i + 1

    new_tag = pad_cat_tag(emotion_set, action_set)  # in new preprocess, remake pad cat tag to split last sent
    '''
    new_tag representation 0 = start_tag, 29 = stop_tag, 30 = pad_tag

    tensor([  0,   2,   1,   2,   1,   2,   1,   2,   1,   2,   1,  29])
    tensor([  0,   2,   1,   2,   1,   2,   1,   2,   1,   2,   1,  29])
    tensor([  0,  21,   1,   1,   1,  29,  30,  30,  30,  30,  30,  30])
    tensor([  0,   2,   1,   2,   1,  29,  30,  30,  30,  30,  30,  30])
    tensor([  0,   1,   1,   1,   1,  29,  30,  30,  30,  30,  30,  30])
    tensor([  0,   2,   1,   2,   1,  29,  30,  30,  30,  30,  30,  30])
    tensor([  0,  17,  17,  17,  29,  30,  30,  30,  30,  30,  30,  30])
    tensor([  0,   1,   1,   1,  29,  30,  30,  30,  30,  30,  30,  30])
    tensor([  0,   3,   2,  29,  30,  30,  30,  30,  30,  30,  30,  30])
    tensor([  0,  17,  17,  29,  30,  30,  30,  30,  30,  30,  30,  30])
    tensor([  0,   2,   2,  29,  30,  30,  30,  30,  30,  30,  30,  30])
    '''
    # batch*tag
    new_tag = Variable(new_tag.cuda())  # default requires_grad = false

    #####################################################tag_preprocess

    new, all_seq_len, sentnum_per_batch = pad_batch(
        batch_data)  # in new preprocess, remake pad batch to split last sent
    # batch * sentnum * (word_list+pad) -> new
    # batch * sentnum * (word_list_length) -> all_seq_len
    # batch * (sentnum) -> sentnum_per_batch
    '''
    new[0]
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
    sentnum_per_batch #contain stop tag
    [11, 11, 5, 5, 5, 5, 4, 4, 3, 3, 3, 3, .....]

    '''

    new2 = batch_numerical(new)
    # batch * sent_num * sent_leng * wv -> new2

    sentbatch_len, for_sentmodel = make_batch2sent(new2)

    # batch * sent_num * sent_leng * wv -> all_sent_num(new_batch) * sent_leng * wv
    # for_sentmodel2 -> torch.Size([326, 7, 100])
    # sentbatch_len -> 326

    # for_sentmodel2 -> torch.Size([7, 326, 100])

    pre_crf_gru = sent(for_sentmodel, len(for_sentmodel), all_seq_len)
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

    #################################################sent network
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
    new_dial, dial_leng = pad_dial(last_var)
    # batch * sent_num * wv -> batch * (sent_num + pad) * wv
    # save dial_leng for masking

    '''
    new_dial

    [11,100]
    [11,100]
    [11,100]
    [11,100]
    [11,100]
    .
    .
    '''

    new_dial = torch.transpose(new_dial, 0, 1)

    return new_dial, new_tag, dial_leng