from torch import nn
from torch.autograd import Variable
import torch


def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))


class BigruCrf(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim):
        super(BigruCrf, self).__init__()

        self.gru = nn.GRU(100, 100, bidirectional=True)  # default requires_grad = true
        self.hidden2tag = nn.Linear(200, 31)  # default requires_grad = true
        self.transitions = nn.Parameter(torch.randn(31, 31))  # [a,b] trans from b to a,  requires_grad = true
        self.transitions.data[0, :] = -10000  # all to start
        self.transitions.data[:, 29] = -10000  # stop to all
        self.transitions.data[:, 30] = -10000  # pad to all
        self.transitions.data[30][30] = 0  # pad to pad
        self.transitions.data[29][30] = 0  # stop to pad

        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix

    def init_hidden(self, batch):
        return Variable(torch.zeros(2, batch, 100).cuda())  # default requires_grad = false

    def _get_gru_features(self, batch, sentence_set):

        hidden = self.init_hidden(batch)
        gru_out, hidden = self.gru(sentence_set, hidden)

        gru_feats = self.hidden2tag(gru_out)

        return gru_feats

    def for_score(self, pre_mask, feats, batch_size):

        score = Variable(torch.zeros((batch_size, 31)).fill_(-10000.).cuda())  # default requires_grad = false
        score[:, self.tag_to_ix['start_tag']] = 0.  # start to all is 0

        mask = Variable(torch.Tensor(pre_mask).cuda())  # default requires_grad = false

        for t in range(feats.size(0)):  # 안에서 연산하는데이터들은 batch*featuresize*featuresize

            mask_t = mask[:, t].unsqueeze(-1).expand_as(score)  # batch_size -> batch_size*featuresize

            score_t = score.unsqueeze(1).expand(-1, *self.transitions.size())  # batch_size*f -> batch_size*f*f

            emit = feats[t].unsqueeze(-1).expand_as(score_t)  # b*f-> b*f*f

            trans = self.transitions.unsqueeze(0).expand_as(score_t)  # b*f*f

            score_t = log_sum_exp(score_t + emit + trans)

            score = score_t * mask_t + score * (1 - mask_t)  # no updating in masked score,all b*f

        score = log_sum_exp(score)
        return score

    def cal_score(self, mask, feats, tag, batch_size):
        score = Variable(torch.FloatTensor(batch_size).fill_(0.).cuda())  # default requires_grad = false

        temp_tag = Variable(tag.cuda())  # default requires_grad = false
        mask_tensor = torch.transpose(torch.FloatTensor(mask), 0, 1)  # seq*batch
        mask_tensor = Variable(mask_tensor.cuda())  # default requires_grad = false

        for i, feat in enumerate(feats):  # seq*batch*feat->batch*feat

            transit = torch.cat(
                [torch.tensor([self.transitions[temp_tag[batch][i + 1], temp_tag[batch][i]]]) for batch in
                 range(batch_size)])

            transit = transit.cuda()

            transit = transit * mask_tensor[i]  # batch*batch->batch

            emit = torch.cat([feat[batch][temp_tag[batch][i + 1]].view(1, -1) for batch in range(batch_size)]).squeeze(
                1)

            emit = emit * mask_tensor[i]  # batch*batch->batch

            score = score + transit + emit

        return score

    def neg_log_likelihood(self, mask, sentence, tags, batch_size):

        feats = self._get_gru_features(batch_size, sentence)

        forward_score = self.for_score(mask, feats, batch_size)

        gold_score = self.cal_score(mask, feats, tags, batch_size)
        '''
        newt = self.transitions.data.cpu().numpy()
        newt[0,:] = 0
        newt[:,29] = 0
        newt[:,30] = 0
        x = np.tile(np.arange(1, 32), (31, 1))
        y = x.transpose()
        z = newt #for visdom
        print(z)


        x = np.tile(np.arange(1, 32), (31, 1))
        y = x.transpose()
        z = (x + y)/20

        # surface
        viz.surf(X=z, opts=dict(colormap='Hot'))
        '''
        return forward_score - gold_score

    def _viterbi_decode(self, mask, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = Variable(torch.full((1, 31), -10000.).cuda())  # default requires_grad = false
        init_vvars[0][0] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for i, feat in enumerate(feats):
            if mask[i] == 0:
                # print('breaked')
                break
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(31):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # terminal_var = forward_var  + self.transitions[self.tag_to_ix['stop_tag']]
        best_tag_id = argmax(forward_var)  # not terminal_var
        path_score = forward_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == 0  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, batch, dummy_input, seq):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_gru_features(batch, dummy_input[1])

        lstm_feats = torch.transpose(lstm_feats, 0, 1)
        mask = dummy_input[0]
        # print(self.transitions)
        # Find the best path, given thue features.
        score, tag_seq = self._viterbi_decode(mask[seq], lstm_feats[seq])
        return score, tag_seq


class Bigru(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim):
        super(Bigru, self).__init__()

        self.gru = nn.GRU(100, 100, bidirectional=True)  # default requires_grad = true
        self.hidden2tag = nn.Linear(200, 31)  # default requires_grad = true

        self.tag_to_ix = tag_to_ix

    def _get_gru_features(self, batch, sentence_set):
        hidden = self.init_hidden(batch)
        gru_out, hidden = self.gru(sentence_set, hidden)

        gru_feats = self.hidden2tag(gru_out)

        return gru_feats

    def forward(self, batch, dummy_input, seq):  # dont confuse this with _forward_alg above.
        feats = self._get_gru_features(batch, sentence)

        return feats


class Linear(nn.Module):
    def __init__(self, tag_size, hidden_dim):
        super(Linear, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_size = tag_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)  # default requires_grad = true

    def forward(self, _input):  # dont confuse this with _forward_alg above.
        feats = self.hidden2tag(_input)
        return feats


class Crf(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim):
        super(Crf, self).__init__()

        self.gru = nn.GRU(100, 100, bidirectional=True)  # default requires_grad = true
        self.hidden2tag = nn.Linear(100, 31)  # default requires_grad = true
        self.transitions = nn.Parameter(torch.randn(31, 31))  # [a,b] trans from b to a,  requires_grad = true
        self.transitions.data[0, :] = -10000  # all to start
        self.transitions.data[:, 29] = -10000  # stop to all
        self.transitions.data[:, 30] = -10000  # pad to all
        self.transitions.data[30][30] = 0  # pad to pad
        self.transitions.data[29][30] = 0  # stop to pad

        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, 100).cuda())  # default requires_grad = false

    def _get_gru_features(self, batch_size, sentence_set):

        hidden = self.init_hidden(batch_size)
        # gru_out, hidden = self.gru(sentence_set, hidden)

        gru_feats = self.hidden2tag(sentence_set)

        return gru_feats

    def for_score(self, pre_mask, feats, batch_size):
        score = Variable(torch.zeros((batch_size, 31)).fill_(-10000.).cuda())  # default requires_grad = false
        score[:, self.tag_to_ix['start_tag']] = 0.  # start to all is 0

        mask = Variable(torch.Tensor(pre_mask).cuda())  # default requires_grad = false

        for t in range(feats.size(0)):  # 안에서 연산하는데이터들은 batch*featuresize*featuresize

            mask_t = mask[:, t].unsqueeze(-1).expand_as(score)  # batch_size -> batch_size*featuresize

            score_t = score.unsqueeze(1).expand(-1, *self.transitions.size())  # batch_size*f -> batch_size*f*f

            emit = feats[t].unsqueeze(-1).expand_as(score_t)  # b*f-> b*f*f

            trans = self.transitions.unsqueeze(0).expand_as(score_t)  # b*f*f

            score_t = log_sum_exp(score_t + emit + trans)

            score = score_t * mask_t + score * (1 - mask_t)  # no updating in masked score,all b*f

        score = log_sum_exp(score)
        return score

    def cal_score(self, mask, feats, tag, batch_size):
        score = Variable(torch.FloatTensor(batch_size).fill_(0.).cuda())  # default requires_grad = false

        temp_tag = Variable(tag.cuda())  # default requires_grad = false
        mask_tensor = torch.transpose(torch.FloatTensor(mask), 0, 1)  # seq*batch
        mask_tensor = Variable(mask_tensor.cuda())  # default requires_grad = false

        for i, feat in enumerate(feats):  # seq*batch*feat->batch*feat

            transit = torch.cat(
                [torch.tensor([self.transitions[temp_tag[batch][i + 1], temp_tag[batch][i]]]) for batch in
                 range(batch_size)])

            transit = transit.cuda()

            transit = transit * mask_tensor[i]  # batch*batch->batch

            emit = torch.cat([feat[batch][temp_tag[batch][i + 1]].view(1, -1) for batch in range(batch_size)]).squeeze(
                1)

            emit = emit * mask_tensor[i]  # batch*batch->batch

            score = score + transit + emit

        return score

    def neg_log_likelihood(self, mask, sentence, tags, batch_size):

        feats = self._get_gru_features(batch_size, sentence)

        forward_score = self.for_score(mask, feats, batch_size)

        gold_score = self.cal_score(mask, feats, tags, batch_size)
        '''
        newt = self.transitions.data.cpu().numpy()
        newt[0,:] = 0
        newt[:,29] = 0
        newt[:,30] = 0
        x = np.tile(np.arange(1, 32), (31, 1))
        y = x.transpose()
        z = newt #for visdom
        print(z)


        x = np.tile(np.arange(1, 32), (31, 1))
        y = x.transpose()
        z = (x + y)/20

        # surface
        viz.surf(X=z, opts=dict(colormap='Hot'))
        '''
        return forward_score - gold_score

    def _viterbi_decode(self, mask, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = Variable(torch.full((1, 31), -10000.).cuda())  # default requires_grad = false
        init_vvars[0][0] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for i, feat in enumerate(feats):
            if mask[i] == 0:
                print('breaked')
                break
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(31):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # terminal_var = forward_var  + self.transitions[self.tag_to_ix['stop_tag']]
        best_tag_id = argmax(forward_var)  # not terminal_var
        path_score = forward_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == 0  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, batch, dummy_input, seq):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_gru_features(batch, dummy_input[1])

        lstm_feats = torch.transpose(lstm_feats, 0, 1)
        mask = dummy_input[0]
        # print(self.transitions)
        # Find the best path, given thue features.
        score, tag_seq = self._viterbi_decode(mask[seq], lstm_feats[seq])
        return score, tag_seq
