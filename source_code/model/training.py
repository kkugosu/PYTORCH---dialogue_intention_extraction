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

from model import SentGru
from model import Encoder
from model import Decoder
from padding import pad_tag, pad_text
from dataset_loader import batchload

BATCH_SIZE = 128
HIDDEN_SIZE = 100
device = torch.device("cuda")

tag_to_ix = {'start_tag': 0, 'stop_tag': 29, 'pad_tag': 30}

working_path = '/home/jongsu/jupyter/pytorch_dialogue_ie/'
WV_PATH = '/home/jongsu/jupyter/pytorch_dialogue_ie/parameter/dialogue_wv'

wv_model = word2vec.Word2Vec(size=100, window=5, min_count=5, workers=4)
wv_model = word2vec.Word2Vec.load(WV_PATH)


encoder1 = Encoder(HIDDEN_SIZE,
                   is_tag_=False,
                   tag_size=31,  # ???
                   bidir=True
                   ).cuda()

encoder2 = Encoder(HIDDEN_SIZE,
                   is_tag_=True,
                   tag_size=31,
                   bidir=True
                   ).cuda()

decoder1 = Decoder(HIDDEN_SIZE,
                   bidir=True
                   ).cuda()

sent = SentGru(HIDDEN_SIZE, True)
print("aaa")



pp = 0
while pp < 1000:
    pp = pp + 1
    batchnum = 1
    for batch_data in batchload(train_data, repeat=False, batchsize=BATCH_SIZE, data_seq=dataseq):
        # load txt data from jsonfile
        batchnum = batchnum + 1
        if batchnum < 47:
            continue

        '''
        batch_data[0].labels_1
        ['0 0 0 0 0 0 0 0 0 0 0 0 4 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 4']
        ['1 2 3 4 1 2 2 2 2 2 1 1 1 2 1 3 4 2 1 2 1 2 1 2 1 1 1 2 2 2 1 3 4 1 1']

        '''
        print('new_batch----------------')

        # print(batch_data[0].Text)

        print("sent_maxlen = ", dialogue_maxlen_per_batch(BATCH_SIZE, batch_data))
        print("dial_len_range = ", len(batch_data[0].Text), " - ", len(batch_data[99].Text))

        sent_to_vec.zero_grad()
        decoder1.zero_grad()
        encoder1.zero_grad()


        en_tag, de_tag = pad_tag(batch_data)
        en_text, en_len, de_text, de_len = pad_text(batch_data, sent_to_vec)

        '''
        coder_mask(en_len)
        [[0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         [0. 0. 0. ... 0. 0. 1.]
         ...
         [0. 0. 0. ... 1. 0. 0.]
         [0. 0. 0. ... 1. 0. 0.]
         [0. 0. 0. ... 1. 0. 0.]]
        torch.sum(torch.tensor(encoder_mask(en_len)),1) 
        1111111111...

        '''

        decoder_hidden = encoder2(en_text, en_tag, coder_mask(en_len, en_len[0], True), de_tag)  # 2,100,100
        # print("encoding compl")

        decoder_sent = 0
        all_output = Variable(torch.zeros(BATCH_SIZE, 1, HIDDEN_SIZE, device=device))
        out = Variable(torch.zeros(BATCH_SIZE, 1, HIDDEN_SIZE, device=device))
        seq = 0

        while (decoder_sent < dialogue_maxlen_per_batch(BATCH_SIZE, batch_data)):  # all_output = seq*batch*hidden
            out, decoder_hidden = decoder1(out, decoder_hidden)
            decoder_sent = decoder_sent + 1

            if seq != 0:
                all_output = torch.cat((all_output, out), 1)

            else:
                all_output = out

            seq = seq + 1
        # print("decoding compl")

        de_mask = coder_mask(de_len, dialogue_maxlen_per_batch(BATCH_SIZE, batch_data), False)  # 100,44
        demask = torch.tensor(de_mask)
        demask = torch.unsqueeze(demask, 2).type(torch.cuda.FloatTensor)

        all_output = torch.mul(demask, all_output)  #

        targetwv = makewv(de_text)

        targetwv = torch.tensor(targetwv).type(torch.cuda.FloatTensor)

        loss = nn.MSELoss()

        mseloss = loss(all_output, targetwv)
        # print("calc loss compl")
        print("loss = ", mseloss)
        mseloss.backward()

        # print("backward compl")
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        # print("optimizer")
        # torch.save(encoder2.state_dict(),working_path + 'parameter/coder/encoder2.pth')
        # torch.save(decoder1.state_dict(),working_path + 'parameter/coder/decoder1.pth')
        # torch.save(sent_to_vec.state_dict(),working_path + 'parameter/coder/sent_to_vec.pth')

        # print("save compl")

        if batchnum == 50:
            break
        '''


        shared_model.zero_grad()
        comp_model.zero_grad()

        en_text, en_tag, en_len = all_preprocess(shared_model, batch_data) #we need split batch_data
        #load batch* (dialogue_length*sent_vec(float)) -> en_text
        #load batch* tag -> en_tag
        #load batch* en_len

        loss = comp_model.neg_log_likelihood(make_mask(en_len), en_text, en_tag, BATCH_SIZE)
        loss,newary_ = loss_filtering(loss,filtering_value, newary_,k)
        batch_loss = torch.sum(loss)
        batch_loss.backward(retain_graph=False)
        optimizer1.step()
        optimizer2.step()

        unuselist = [en_text, en_tag, en_len]
        del unuselist
        '''
