import torch
from torch import nn


class SentGru(nn.Module):
    def __init__(self, hidden_size, bidirectional, device):
        super(SentGru, self).__init__()
        self.hidden_size = hidden_size
        self.bid = bidirectional
        self.device = device
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
        return torch.zeros(2, batch_size, self.hidden_size, device=self.device, requires_grad=False)

    def masking_f(self, remake, all_seq_len):

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