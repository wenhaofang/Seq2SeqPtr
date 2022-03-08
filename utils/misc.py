import torch

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

def get_inp_from_batch(batch, device, is_copy, is_coverage):
    enc_inp = Variable(torch.from_numpy(batch.enc_inp).long())
    enc_len = Variable(torch.from_numpy(batch.enc_len).long())
    enc_pad_mask = Variable(torch.from_numpy(batch.enc_pad_mask).float())

    enc_inp = enc_inp.to(device)
    enc_len = enc_len.to(device)
    enc_pad_mask = enc_pad_mask.to(device)

    extra_zeros = None
    enc_inp_extend_vocab = None
    if  is_copy:
        enc_inp_extend_vocab = Variable(torch.from_numpy(batch.enc_inp_extend_vocab).long())
        enc_inp_extend_vocab = enc_inp_extend_vocab.to(device)

        batch_size  = len(batch.enc_len)
        max_oov_len = max([len(oov) for oov in batch.oovs])
        if  max_oov_len > 0:
            extra_zeros = Variable(torch.zeros((batch_size, max_oov_len)))
            extra_zeros = extra_zeros.to(device)

    coverage = None
    if  is_coverage:
        coverage = Variable(torch.zeros(enc_inp.shape))
        coverage = coverage.to(device)

    return enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage

def get_out_from_batch(batch, device):
    dec_inp = Variable(torch.from_numpy(batch.dec_inp).long())
    dec_out = Variable(torch.from_numpy(batch.dec_out).long())
    dec_len = Variable(torch.from_numpy(batch.dec_len).long())
    dec_pad_mask = Variable(torch.from_numpy(batch.dec_pad_mask).float())

    dec_inp = dec_inp.to(device)
    dec_out = dec_out.to(device)
    dec_len = dec_len.to(device)
    dec_pad_mask = dec_pad_mask.to(device)

    return dec_inp, dec_len, dec_pad_mask, dec_out

def train(batch, model, device, optimizer, is_copy, is_coverage, cov_loss_wt, max_dec_step, grad_clip):

    model.train()

    enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage \
        = get_inp_from_batch(batch, device, is_copy, is_coverage)
    dec_inp, dec_len, dec_pad_mask, dec_out \
        = get_out_from_batch(batch, device)

    loss, cove_loss = model(
        enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage,
        dec_inp, dec_len, dec_pad_mask, dec_out, cov_loss_wt, max_dec_step
    )

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), cove_loss
