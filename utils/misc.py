import os
import torch

from torch.autograd import Variable

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def save_sample(folder, sources, targets, predict):
    sources_path = os.path.join(folder, 'sources.txt')
    targets_path = os.path.join(folder, 'targets.txt')
    predict_path = os.path.join(folder, 'predict.txt')
    for data, file_path in zip (
        [sources, targets, predict],
        [sources_path, targets_path, predict_path]
    ):
        with open(file_path, 'w', encoding = 'utf-8') as txt_file:
            txt_file.writelines([seq + '\n' for seq in data])

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), cove_loss

def valid(batch, model, device, is_copy, is_coverage, cov_loss_wt, max_dec_step):

    model.eval()

    enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage \
        = get_inp_from_batch(batch, device, is_copy, is_coverage)
    dec_inp, dec_len, dec_pad_mask, dec_out \
        = get_out_from_batch(batch, device)

    loss, cove_loss = model(
        enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage,
        dec_inp, dec_len, dec_pad_mask, dec_out, cov_loss_wt, max_dec_step
    )

    return loss.item(), cove_loss

def test (batch, model, device, is_copy, is_coverage, vocab, min_dec_step, max_dec_step, beam_width):

    model.eval()

    UNK_TOKEN = vocab.special['UNK_TOKEN']
    PAD_TOKEN = vocab.special['PAD_TOKEN']
    BOS_TOKEN = vocab.special['BOS_TOKEN']
    EOS_TOKEN = vocab.special['EOS_TOKEN']

    UNK_ID = vocab.convert_word_to_id(UNK_TOKEN)
    PAD_ID = vocab.convert_word_to_id(PAD_TOKEN)
    BOS_ID = vocab.convert_word_to_id(BOS_TOKEN)
    EOS_ID = vocab.convert_word_to_id(EOS_TOKEN)

    enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage \
        = get_inp_from_batch(batch, device, is_copy, is_coverage)

    result = model.predict(
        enc_inp, enc_len, enc_pad_mask, enc_inp_extend_vocab, extra_zeros, coverage,
        min_dec_step, max_dec_step, BOS_ID, EOS_ID, UNK_ID, beam_width
    )

    src = batch.src[0]
    trg = batch.trg[0]

    oovs = batch.oovs[0] if is_copy else None

    pred = []
    for word_id in result[0]:
        if  word_id >= vocab.size:
            if  oovs:
                word = oovs[vocab.size - word_id]
            else:
                raise Exception('word_id out of range')
        else:
            if  word_id not in [UNK_ID, PAD_ID, BOS_ID, EOS_ID]:
                word = vocab.convert_id_to_word(word_id)
            else:
                continue

        pred.append(word)

    pred = ' '.join(pred)

    return src, trg, pred
