import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import  pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder   = nn.LSTM     (emb_dim, hid_dim, batch_first = True, bidirectional = True)
        self.transform = nn.Linear   (hid_dim * 2 , hid_dim * 2 , bias = False)

    def forward(self, seq, seq_len):
        '''
        Params:
            seq    : (batch_size, seq_len)
            seq_len: (batch_size)
        Return:
            outputs: (batch_size , seq_len , hid_dim * 2)
            feature: (batch_size * seq_len , hid_dim * 2)
            hiddens: tuple (
                (2, batch_size, hid_dim),
                (2, batch_size, hid_dim)
            )
        '''
        embedded = self.dropout(self.embedding(seq))
        embedded = pack_padded_sequence(embedded, seq_len.to('cpu'), batch_first = True, enforce_sorted = False)
        outputs, hiddens = self.encoder(embedded)
        outputs, lengths = pad_packed_sequence(outputs, batch_first = True)
        feature = self.transform(outputs.reshape(-1, self.hid_dim * 2))
        return outputs, feature, hiddens

class ReduceState(nn.Module):
    def __init__(self, hid_dim):
        super(ReduceState, self).__init__()
        self.hid_dim = hid_dim
        self.transformH = nn.Linear(hid_dim * 2, hid_dim)
        self.transformC = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, hiddens):
        '''
        Params:
            hiddens: tuple (
                (2, batch_size, hid_dim),
                (2, batch_size, hid_dim)
            )
        Return:
            hiddens: tuple (
                (1, batch_size, hid_dim),
                (1, batch_size, hid_dim)
            )
        '''
        h_r = hiddens[0]
        c_r = hiddens[1]
        h_i = h_r.transpose(0, 1).reshape(-1, self.hid_dim * 2)
        c_i = c_r.transpose(0, 1).reshape(-1, self.hid_dim * 2)
        h_o = F.relu(self.transformH(h_i)).unsqueeze(0)
        c_o = F.relu(self.transformH(c_i)).unsqueeze(0)
        return h_o, c_o

class Attention(nn.Module):
    def __init__(self, hid_dim, is_coverage):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.is_coverage = is_coverage
        self.dec_fc = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.att_fc = nn.Linear(hid_dim * 2, 1, bias = False)
        if  is_coverage:
            self.cov_fc = nn.Linear(1, hid_dim * 2, bias = False)

    def forward(self, s_t, enc_out, enc_fea, enc_pad_mask, coverage):
        '''
        Params:
            s_t: (batch_size, hid_dim * 2)
            enc_out: (batch_size , seq_len, hid_dim * 2)
            enc_fea: (batch_size * seq_len, hid_dim * 2)
            enc_pad_mask: (batch_size, seq_len)
            coverage    : (batch_size, seq_len)
        Return:
            c_t: (batch_size, hid_dim * 2)
            attn    : (batch_size, seq_len)
            coverage: (batch_size, seq_len)
        '''
        batch_size, seq_len, fea_dim = enc_out.shape

        dec_fea = self.dec_fc(s_t)
        dec_fea = dec_fea.unsqueeze(1).expand(batch_size, seq_len, fea_dim).reshape(-1, fea_dim)

        att_fea = enc_fea + dec_fea

        if  self.is_coverage:
            cov_inp = coverage.view(-1, 1)
            cov_fea = self.cov_fc(cov_inp)

            att_fea = att_fea + cov_fea

        scores = self.att_fc(torch.tanh(att_fea))
        scores = scores.view(-1, seq_len)

        attn = F.softmax(scores, dim = 1) * enc_pad_mask
        norm = torch.sum(attn, dim = 1 , keepdim = True)
        attn = attn / norm

        c_t = torch.bmm(attn.unsqueeze(1), enc_out)
        c_t = c_t.view (-1, self.hid_dim * 2)

        if  self.is_coverage:
            coverage = coverage + attn

        return c_t, attn, coverage

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout, is_copy, is_coverage):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.is_copy = is_copy
        self.dropout   = nn.Dropout  (dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transform = nn.Linear   (hid_dim * 2 + emb_dim, emb_dim)
        self.decoder   = nn.LSTM     (emb_dim, hid_dim, batch_first = True, bidirectional = False)
        self.attention = Attention   (hid_dim, is_coverage)
        if  is_copy:
            self.p_gen_fc = nn.Linear(hid_dim * 4 + emb_dim, 1)
        self.fc1 = nn.Linear(hid_dim * 3, hid_dim)
        self.fc2 = nn.Linear(hid_dim , vocab_size)

    def forward( self,
        y_t, s_t, c_t, enc_out, enc_fea, enc_pad_mask,
        enc_inp_extend_vocab, extra_zeros, coverage, t
    ):
        '''
        Params:
            y_t: (batch_size)
            s_t: tuple(
                (1, batch_size, hid_dim),
                (1, batch_size, hid_dim)
            )
            c_t: (batch_size, hid_dim * 2)
            enc_out     : (batch_size , seq_len, hid_dim * 2)
            enc_fea     : (batch_size * seq_len, hid_dim * 2)
            enc_pad_mask: (batch_size , seq_len)
            enc_inp_extend_vocab: (batch_size, seq_len)
            extra_zeros: (batch_size, max_oov_len)
            coverage   : (batch_size, seq_len)
            t: Int, current time step
        Return:
            final_prob: (batch_size, vocab_size + max_oov_len)
            s_t: tuple(
                (1, batch_size, hid_dim),
                (1, batch_size, hid_dim)
            )
            c_t: (batch_size, hid_dim * 2)
            extra_prob   : (batch_size, seq_len)
            coverage_next: (batch_size, seq_len)
        '''
        if  not self.training and t == 0:
            dec_h = s_t[0]
            dec_c = s_t[1]
            s_t_hat = torch.cat((
                dec_h.view(-1, self.hid_dim),
                dec_c.view(-1, self.hid_dim),
            ) , dim = 1)

            c_t, extra_prob, coverage_next \
                = self.attention(s_t_hat, enc_out, enc_fea, enc_pad_mask, coverage)

            coverage = coverage_next

        y_t_emb = self.dropout(self.embedding(y_t))

        x = torch.cat((c_t, y_t_emb), dim = 1)
        x = self.transform(x)

        dec_out , s_t = self.decoder(x.unsqueeze(1), s_t)
        dec_out = dec_out.view(-1,self.hid_dim)

        dec_h = s_t[0]
        dec_c = s_t[1]
        s_t_hat = torch.cat((
            dec_h.view(-1, self.hid_dim),
            dec_c.view(-1, self.hid_dim),
        ) , dim = 1)

        c_t, extra_prob, coverage_next \
            = self.attention(s_t_hat, enc_out, enc_fea, enc_pad_mask, coverage)

        p_gen = None
        if  self.is_copy:
            p_gen = torch.cat((c_t, s_t_hat, x), dim = 1)
            p_gen = self.p_gen_fc(p_gen)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((dec_out, c_t), dim = 1)
        output = self.fc1(output)
        output = self.fc2(output)

        vocab_prob = F.softmax(output, dim = 1)

        if  self.is_copy:
            vocab_prob_ = vocab_prob * (p_gen)
            extra_prob_ = extra_prob * (1 - p_gen)

            if  extra_zeros is not None:
                vocab_prob_ = torch.cat((vocab_prob_, extra_zeros), dim = 1)

            final_prob = vocab_prob_.scatter_add(1, enc_inp_extend_vocab, extra_prob_)

        else:
            final_prob = vocab_prob

        return final_prob, s_t, c_t, extra_prob, coverage_next

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout, is_copy, is_coverage):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, emb_dim, hid_dim, dropout)
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dropout, is_copy, is_coverage)
        self.reduce_state = ReduceState(hid_dim)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.is_copy = is_copy
        self.is_coverage = is_coverage

    def forward( self,
        enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage,
        dec_inp, dec_len, dec_pad_mask, dec_out, cov_loss_wt, max_dec_step
    ):
        enc_out, enc_fea, enc_h = self.encoder(enc_inp, enc_len)

        s_t = self.reduce_state(enc_h)

        c_t = Variable(torch.zeros((enc_out.shape[0], self.hid_dim * 2)).to(enc_out.device))

        step_losses = []
        cove_losses = []
        for t in range(min(max_dec_step, torch.max(dec_len).item())):
            y_t = dec_inp[:, t]
            z_t = dec_out[:, t]
            final_prob, s_t, c_t, attn, coverage_next = self.decoder(
                y_t, s_t, c_t, enc_out, enc_fea, enc_pad_mask,
                enc_inp_extend_vocab, extra_zeros, coverage, t
            )

            step_mask = dec_pad_mask[:, t]
            step_loss = - torch.log (
                torch.gather(final_prob, 1, z_t.unsqueeze(1)).squeeze(1) + 1e-12
            )

            if  self.is_coverage:
                cove_loss = torch.sum(torch.min(attn, coverage), dim = 1)
                cove_losses.append(cove_loss * step_mask)

                coverage = coverage_next
                step_loss = step_loss + cov_loss_wt * cove_loss

            step_losses.append(step_loss * step_mask)

        step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_step_loss = torch.mean(step_losses / dec_len)

        if  self.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = torch.mean(cove_losses / dec_len)

            return batch_step_loss, batch_cove_loss.item()

        return batch_step_loss, 0.

def get_module(option, vocab_size):
    return Seq2Seq(
        vocab_size,
        option.emb_dim,
        option.hid_dim,
        option.dropout,
        option.is_copy,
        option.is_coverage
    )

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    vocab_size = 50000

    module = get_module(option, vocab_size)

    batch_size = option.batch_size
    enc_step = option.max_enc_step
    dec_step = option.max_dec_step

    max_oov_len = 9

    loss, cove_loss = module(
        enc_inp              = torch.randint(0, vocab_size, (batch_size, enc_step)),
        enc_len              = torch.tensor (list(range(enc_step, enc_step - batch_size, -1))),
        enc_pad_mask         = torch.ones((batch_size, enc_step)),
        enc_inp_extend_vocab = torch.randint(0, vocab_size + max_oov_len, (batch_size, enc_step)),
        extra_zeros          = torch.zeros((batch_size, max_oov_len)),
        coverage             = torch.zeros((batch_size, enc_step)),
        dec_inp              = torch.randint(0, vocab_size, (batch_size, dec_step)),
        dec_len              = torch.tensor(list(range(dec_step, dec_step - batch_size, -1))),
        dec_pad_mask         = torch.ones((batch_size, dec_step)),
        dec_out              = torch.randint(0, vocab_size, (batch_size, dec_step)),
        cov_loss_wt          = option.cov_loss_wt,
        max_dec_step         = option.max_dec_step
    )

    print(loss, cove_loss)
