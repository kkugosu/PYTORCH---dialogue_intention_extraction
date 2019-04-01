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



class SentGru(nn.Module):
    def __init__(self, hidden_size, bidirectional):
        super(SentGru, self).__init__()
        self.hidden_size = hidden_size
        self.bid = bidirectional
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=self.bid)
        self.lastnet = nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, char, batch_size, masking_v):
        h0 = self.init_hidden(batch_size)
        # char 7,326,100 [6][0] = 0000....<pad>
        gru_out, h0 = self.gru(char, h0)
        # gru 7,326,200

        last_hidden_state = self.masking_f(gru_out, masking_v)
        # [326,200]

        last_w = self.lastnet(last_hidden_state)
        # [326,100]

        return last_w

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.bidirectional, self.hidden_size, device=device, requires_grad=False)

    def masking_f(self, new_sent, all_seq_len):
        remake = new_sent
        # remake [326, 7, 200]
        i = 0
        while i < len(remake):
            if i == 0:
                # all_seq_len[0]-1 -> 5
                tensor = remake[0][all_seq_len[0] - 1].view(1, 2*self.hidden_size)
            else:
                tensor = torch.cat((tensor, remake[i][all_seq_len[i] - 1].view(1, 2*self.hidden_size)), 0)
            i = i + 1
        return tensor


class Encoder(nn.Module):
    def __init__(self, hidden_state_size, is_tag_, tag_size, bidir):
        super(Encoder, self).__init__()
        self.h_size = hidden_state_size
        self.is_tag = is_tag_
        self.t_size = tag_size
        self.bid = bidir
        self.gru = nn.GRU(self.h_size, self.h_size, batch_first=True, bidirectional=self.bid)
        self.embed_tag = nn.Embedding(self.t_size, self.h_size)
        self.combinput = nn.Linear(2 * self.h_size, self.h_size)
        self.comblast_t = nn.Linear(3 * self.h_size, 2 * self.h_size)

    def forward(self, input_, input_tag, mask, last_tag):
        hidden = self.init_hidden()
        mask = Variable(torch.tensor(mask).cuda())
        last_tag = Variable(torch.tensor(last_tag).cuda())
        if not self.is_tag:
            input_ = torch.transpose(input_, 0, 1)
            output, hidden_state = self.gru(Variable(input_.cuda()), hidden)
        else:
            emb_tag = self.embed_tag(Variable(input_tag.cuda()))

            newinput = torch.cat((emb_tag, input_), 2)  # 128,3,200

            input_ = self.combinput(newinput)

            output, hidden_state = self.gru(input_, hidden)  # outputsize [3, 128, 200] hiddensize [2, 128, 100]
            len_info = torch.tensor(mask)

            len_info = torch.unsqueeze(len_info, 2).type(torch.cuda.FloatTensor)  # 128,3,1 001001001...

            new_output = torch.mul(output, len_info)  # 128,3,200

            new_output = torch.sum(new_output, 1)  # 128,200

            new_last_tag = self.embed_tag(last_tag)  # 128,100

            cat = torch.cat((new_output, new_last_tag), 1)  # 128,300

            lastoutput = self.comblast_t(cat)  # 128,200
            lastoutput = torch.reshape(lastoutput, (BATCH_SIZE, 2, HIDDEN_SIZE))
            lastoutput = torch.transpose(lastoutput, 0, 1).contiguous()

            # not hidden state. output post processing need
        return lastoutput

    def init_hidden(self):
        return Variable(torch.zeros(2, BATCH_SIZE, HIDDEN_SIZE, device=device))


class Decoder(nn.Module):  # target padding sos
    def __init__(self, hidden_state_size, bidir):
        super(Decoder, self).__init__()
        self.bid = bidir
        self.h_size = hidden_state_size
        self.comboutput = nn.Linear(2 * self.h_size, self.h_size)
        self.gru = nn.GRU(self.h_size, self.h_size, batch_first=True, bidirectional=self.bid)

    def forward(self, new_input, hidden_state):
        output, hidden_state = self.gru(new_input, hidden_state)
        output = self.comboutput(output)
        return output, hidden_state

    def init_hidden(self):
        return Variable(torch.zeros(2, BATCH_SIZE, HIDDEN_SIZE, device=device))


