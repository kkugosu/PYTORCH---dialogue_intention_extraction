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

from model import SentGru, Encoder, Decoder
from padding import pad_tag, pad_text
from cal_maxlen import sentence_maxlen_per_dialogue, sentence_maxlen_per_batch, dialogue_maxlen_per_batch
from mask import coder_mask
from decode import decoding
from dataset_loader import batchload, MyTabularDataset


BATCH_SIZE = 128
HIDDEN_SIZE = 100
device = torch.device("cuda")

tag_to_ix = {'start_tag': 0, 'stop_tag': 29, 'pad_tag': 30}

working_path = '/home/jongsu/jupyter/pytorch_dialogue_ie/'
WV_PATH = '/home/jongsu/jupyter/pytorch_dialogue_ie/parameter/dialogue_wv'

wv_model = word2vec.Word2Vec(size=100, window=5, min_count=5, workers=4)
wv_model = word2vec.Word2Vec.load(WV_PATH)

my_fields = {'dial': ('Text', data.Field(sequential=True)),
             'emo': ('labels_1', data.Field(sequential=False)),
             'act': ('labels_2', data.Field(sequential=False))}
print("make data")
train_data = MyTabularDataset.splits(path=working_path, train='data_jsonfile/full_data.json', fields=my_fields)
train_data = sorted(train_data, key=lambda x: sentence_maxlen_per_dialogue(x))
train_data = train_data[:-5118]  # exclude dialogue which has extremely long sentence (0~11117 => 0~9999)
train = sorted(train_data, key=lambda x: -len(x.Text))  # reordering training dataset with number of sentences
# low index has much sentence because afterwards we use torch pad_sequence

print("make data")

# dataseq = torch.arange(end = len(train),dtype=torch.int)
dataseq = torch.zeros((len(train)), dtype=torch.int).fill_(5950)

encoder1 = Encoder(HIDDEN_SIZE,
                   is_tag_=False,
                   tag_size=31,  # ???
                   bidir=True,
                   device=device
                   ).cuda()

encoder2 = Encoder(HIDDEN_SIZE,
                   is_tag_=True,
                   tag_size=31,
                   bidir=True,
                   device=device
                   ).cuda()

decoder1 = Decoder(HIDDEN_SIZE,
                   bidir=True,
                   device=device
                   ).cuda()

sent = SentGru(HIDDEN_SIZE, bidirectional=True, device=device).cuda()

learning_rate = 0
optimizer1 = optim.SGD(decoder1.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer2 = optim.SGD(encoder2.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer3 = optim.SGD(sent.parameters(), lr=learning_rate, weight_decay=1e-4)
print("now ready")
pp = 0
while pp < 10:
    pp = pp + 1
    batchnum = 1
    for batch_data in batchload(train_data, repeat=False, batchsize=BATCH_SIZE, data_seq=dataseq):
        # load txt data from jsonfile
        print('new_batch----------------')
        print("sent_maxlen = ", sentence_maxlen_per_batch(batch_data))
        print("dial_len_range = ", len(batch_data[0].Text), " - ", len(batch_data[99].Text))

        batchnum = batchnum + 1
        #if batchnum < 47:
        #    continue

        sent.zero_grad()
        decoder1.zero_grad()
        encoder1.zero_grad()

        en_tag, de_tag = pad_tag(batch_data, tag_to_ix, device)
        # load batch* tag -> en_tag
        en_text_vec, en_len, de_text_vec, de_len = pad_text(sent, wv_model, batch_data, device)
        # load batch* (dialogue_length*sent_vec(float)) -> en_text_vec
        # load batch* en_len

        de_maxlen = sentence_maxlen_per_batch(batch_data)
        en_maxlen = dialogue_maxlen_per_batch(en_len)
        print("????")
        break
        encoder_mask = coder_mask(en_len, en_maxlen, True)
        decoder_mask = coder_mask(de_len, de_maxlen, False)

        decoder_input = encoder2(en_text_vec, en_tag, encoder_mask, de_tag, BATCH_SIZE)
        decoder_output = decoding(decoder1, decoder_input, de_maxlen, decoder_mask, BATCH_SIZE)

        loss = nn.MSELoss()

        mseloss = loss(decoder_output, de_text_vec)
        print("loss = ", mseloss)
        mseloss.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        # torch.save(encoder2.state_dict(),working_path + 'parameter/coder/encoder2.pth')
        # torch.save(decoder1.state_dict(),working_path + 'parameter/coder/decoder1.pth')
        # torch.save(sent_to_vec.state_dict(),working_path + 'parameter/coder/sent_to_vec.pth')

        if batchnum == 50:
            break
        '''
        unuselist = [en_text, en_tag, en_len]
        del unuselist
        '''
