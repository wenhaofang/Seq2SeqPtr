import os
import time
import queue
import random
import numpy as np
import pandas as pd
import threading

class Vocab():
    def __init__(self, file_path, min_freq, max_numb):
        self.min_freq = min_freq
        self.max_numb = max_numb
        self.size = 0
        self.word2id = {}
        self.id2word = {}
        self.special = {
            'UNK_TOKEN': '[UNK]',
            'PAD_TOKEN': '[PAD]',
            'BOS_TOKEN': '[BOS]',
            'EOS_TOKEN': '[EOS]'
        }
        self.read_data(file_path)

    def read_data(self,file_path):
        self.size = 0
        self.word2id.clear()
        self.id2word.clear()
        for word in self.special.values():
            self.word2id[word] = self.size
            self.id2word[self.size] = word
            self.size += 1
        with open(file_path, 'r', encoding = 'utf-8') as tsv_file:
            for line in tsv_file:
                item = line.split()
                if len(item) != 2:
                    continue
                word = str(item[0])
                freq = int(item[1])
                if  word in self.special.values():
                    raise Exception(
                        'Special word should not be in vocabulary file'
                    )
                if  word in self.word2id.keys():
                    raise Exception(
                        'Duplicated word exists in vocabulary file: %s' % word
                    )
                if  freq < self.min_freq:
                    continue
                if  self.size >= self.max_numb:
                    break
                self.word2id[word] = self.size
                self.id2word[self.size] = word
                self.size += 1

    def convert_word_to_id(self,word):
        if  word not in self.word2id:
            return self.word2id[self.special['UNK_TOKEN']]
        else:
            return self.word2id[word]

    def convert_id_to_word(self,word_id):
        if  word_id not in self.id2word:
            raise ValueError('id not found in vocab: %d' % word_id)
        else:
            return self.id2word[word_id]

class Datum():
    def __init__(self, datum, vocab, max_enc_step, max_dec_step, is_copy):

        self.is_copy = is_copy

        src = datum[0]
        trg = datum[1]

        self.src = src
        self.trg = trg

        src_wds = src.split()
        trg_wds = trg.split()

        src_ids = [vocab.convert_word_to_id(wd) for wd in src_wds]
        trg_ids = [vocab.convert_word_to_id(wd) for wd in trg_wds]

        self.enc_inp , self.enc_len = \
            self.get_enc_seq(src_ids, max_enc_step)

        self.dec_inp , self.dec_len , self.dec_out = \
            self.get_dec_seq(trg_ids, max_dec_step, vocab)

        if  self.is_copy:

            self.enc_inp_extend_vocab, self.oovs = \
                self.src2ids(src_wds[:max_enc_step], vocab)

            dec_inp_extend_vocab = \
                self.trg2ids(trg_wds[:max_dec_step], vocab, self.oovs)

            _, _, self.dec_out = \
                self.get_dec_seq(dec_inp_extend_vocab, max_dec_step, vocab)

    def get_enc_seq(self, seq_ids, max_len):

        seq = seq_ids

        if  len(seq) > max_len:
            seq = seq[:max_len]

        return seq, len(seq)

    def get_dec_seq(self, seq_ids, max_len, vocab):

        src = seq_ids[:]
        trg = seq_ids[:]
        src.insert(0 , vocab.convert_word_to_id(vocab.special['BOS_TOKEN']))
        
        if  len(src) > max_len:
            src = src[:max_len]
            trg = trg[:max_len]
        else:
            trg.append(vocab.convert_word_to_id(vocab.special['EOS_TOKEN']))

        return src, len(src), trg

    def src2ids(self, seq_wds, vocab):

        ids = []
        oov = []
        unk = vocab.convert_word_to_id(vocab.special['UNK_TOKEN'])

        for word in seq_wds:
            word_id = vocab.convert_word_to_id(word)
            if  word_id == unk:
                if word in oov:
                    pass
                else:
                    oov.append(word)
                ids.append(
                    vocab.size + oov.index(word)
                )
            else:
                ids.append(word_id)

        return ids, oov

    def trg2ids(self, seq_wds, vocab, oov):

        ids = []
        unk = vocab.convert_word_to_id(vocab.special['UNK_TOKEN'])

        for word in seq_wds:
            word_id = vocab.convert_word_to_id(word)
            if  word_id == unk:
                if word in oov:
                    ids.append(
                        vocab.size + oov.index(word)
                    )
                else:
                    ids.append(unk)
            else:
                ids.append(word_id)

        return ids

    def pad_enc_seq(self, max_len, pad_idx):

        while len(self.enc_inp) < max_len:
            self.enc_inp.append(pad_idx)

        if not self.is_copy:
            return

        while len(self.enc_inp_extend_vocab) < max_len:
            self.enc_inp_extend_vocab.append(pad_idx)

    def pad_dec_seq(self, max_len, pad_idx):

        while len(self.dec_inp) < max_len:
            self.dec_inp.append(pad_idx)

        while len(self.dec_out) < max_len:
            self.dec_out.append(pad_idx)

class Batch():
    def __init__(self, datas, vocab, batch_size, max_dec_step, is_copy):

        self.is_copy = is_copy

        self.batch_size = batch_size

        self.save_ori_seqs(datas)
        self.init_enc_seqs(
            datas,
            max([datum.enc_len for datum in datas]),
            vocab.convert_word_to_id(vocab.special['PAD_TOKEN'])
        )
        self.init_dec_seqs(
            datas,
            max_dec_step,
            vocab.convert_word_to_id(vocab.special['PAD_TOKEN'])
        )

    def save_ori_seqs(self, data):
        self.src = [datum.src for datum in data]
        self.trg = [datum.trg for datum in data]

    def init_enc_seqs(self, datas, max_len, pad_idx):
        for datum in datas:
            datum.pad_enc_seq(max_len, pad_idx)

        self.enc_inp = np.zeros((self.batch_size, max_len))
        self.enc_len = np.zeros((self.batch_size))
        self.enc_pad_mask = np.zeros((self.batch_size, max_len))

        for i, datum in enumerate(datas):
            self.enc_inp[i] = datum.enc_inp[:]
            self.enc_len[i] = datum.enc_len
            for j in range(datum.enc_len):
                self.enc_pad_mask[i][j] = 1

        if  self.is_copy:
            self.enc_inp_extend_vocab = np.zeros((self.batch_size, max_len))

            for i, datum in enumerate(datas):
                self.enc_inp_extend_vocab[i] = datum.enc_inp_extend_vocab[:]

            self.oovs = [datum.oovs for datum in datas] # nested list

    def init_dec_seqs(self, datas, max_len, pad_idx):
        for datum in datas:
            datum.pad_dec_seq(max_len, pad_idx)

        self.dec_inp = np.zeros((self.batch_size, max_len))
        self.dec_out = np.zeros((self.batch_size, max_len))
        self.dec_len = np.zeros((self.batch_size))
        self.dec_pad_mask = np.zeros((self.batch_size, max_len))

        for i, datum in enumerate(datas):
            self.dec_inp[i] = datum.dec_inp[:]
            self.dec_out[i] = datum.dec_out[:]
            self.dec_len[i] = datum.dec_len
            for j in range(datum.dec_len):
                self.dec_pad_mask[i][j] = 1

class Batcher():

    BATCH_QUEUE_MAX = 100

    def __init__(self, file_path, vocab, batch_size, max_enc_step, max_dec_step, is_copy, single_pass, beam_search):

        self.datas = self.read_data(file_path) # generator
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_enc_step = max_enc_step
        self.max_dec_step = max_dec_step
        self.is_copy = is_copy
        self.single_pass = single_pass
        self.beam_search = beam_search

        self._datum_queue = queue.Queue(self.BATCH_QUEUE_MAX * batch_size)
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)

        if  single_pass:
            self._datum_queue_thread_number = 1 # must be 1
            self._batch_queue_thread_number = 1 # must be 1
            self._bucketing_cache_size = 1      # must be 1
        else:
            self._datum_queue_thread_number = 1
            self._batch_queue_thread_number = 1
            self._bucketing_cache_size = 1

        self._finished_reading = False

        self._datum_queue_threads = []
        for _ in range(self._datum_queue_thread_number):
            self._datum_queue_threads.append(
                threading.Thread(target = self.fill_datum_queue)
            )
            self._datum_queue_threads[-1].daemon = True
            self._datum_queue_threads[-1].start()

        self._batch_queue_threads = []
        for _ in range(self._batch_queue_thread_number):
            self._batch_queue_threads.append(
                threading.Thread(target = self.fill_batch_queue)
            )
            self._batch_queue_threads[-1].daemon = True
            self._batch_queue_threads[-1].start()

        if not single_pass: 
            self._watch_queue_thread = \
                threading.Thread(target = self.watch_threads)
            self._watch_queue_thread.daemon = True
            self._watch_queue_thread.start()

    def read_data(self, file_path):
        datas = []
        df = pd.read_csv(file_path, sep = '\t')
        for row in df.itertuples():
            src = getattr(row, 'article' )
            trg = getattr(row, 'abstract')
            if len(src) > 0 and len(trg) > 0:
                datas.append((src, trg))

        while True:
            if  not self.single_pass:
                random.shuffle(datas)

            for datum in datas:
                yield datum

            if  self.single_pass:
                break

    def fill_datum_queue(self):
        while True:
            try:
                datum = self.datas.__next__()
            except StopIteration:
                if  self.single_pass:
                    self._finished_reading = True
                    break
                else:
                    raise Exception('single_pass is off, but generator is out of data')

            self._datum_queue.put(
                Datum(datum, self.vocab, self.max_enc_step, self.max_dec_step, self.is_copy)
            )

    def fill_batch_queue(self):
        while True:
            if  self.beam_search:
                datum = self._datum_queue.get()
                datas = [datum for _ in range(self.batch_size)] # batch_size is beam_width
                self._batch_queue.put(
                    Batch(datas, self.vocab, self.batch_size, self.max_dec_step, self.is_copy)
                )
            else:
                all_datas = []
                for i in range(self.batch_size * self._bucketing_cache_size):
                    all_datas.append(self._datum_queue.get())

                all_datas = sorted(all_datas, key = lambda datum: datum.enc_len, reverse = True)

                all_batch = []
                for i in range(0, len(all_datas), self.batch_size):
                    all_batch.append(all_datas[i: self.batch_size + i])

                if  not self.single_pass:
                    random.shuffle(all_batch)

                for datas in all_batch:
                    self._batch_queue.put(
                        Batch(datas, self.vocab, self.batch_size, self.max_dec_step, self.is_copy)
                    )

    def watch_threads(self):
        while True:

            time.sleep(60)

            for i, t in enumerate(self._datum_queue_threads):
                if not t.is_alive():
                    new_t = threading.Thread(target = self.fill_datum_queue)
                    self._datum_queue_threads[i] = new_t
                    new_t.daemon = True
                    new_t.start()

            for i, t in enumerate(self._batch_queue_threads):
                if not t.is_alive():
                    new_t = threading.Thread(target = self.fill_batch_queue)
                    self._batch_queue_threads[i] = new_t
                    new_t.daemon = True
                    new_t.start()

    def get_batch(self):
        if (
            self._batch_queue.qsize() == 0 and
            self._finished_reading == True
        ):
            return None

        batch = self._batch_queue.get()
        return batch

def get_vocab(option):
    min_freq = option.min_freq
    max_numb = option.max_numb
    vocab_path = os.path.join(option.targets_path, option.vocab_file)
    return Vocab(vocab_path , min_freq , max_numb)

def get_datas(option, vocab):
    train_path = os.path.join(option.targets_path, option.train_file)
    valid_path = os.path.join(option.targets_path, option.valid_file)
    test_path  = os.path.join(option.targets_path, option.test_file )
    batch_size = option.batch_size
    beam_width = option.beam_width
    max_enc_step = option.max_enc_step
    max_dec_step = option.max_dec_step
    is_copy = option.is_copy
    train_loader = Batcher(train_path, vocab, batch_size, max_enc_step, max_dec_step, is_copy, single_pass = False, beam_search = False)
    valid_loader = Batcher(valid_path, vocab, batch_size, max_enc_step, max_dec_step, is_copy, single_pass = True , beam_search = False)
    test_loader  = Batcher(test_path , vocab, beam_width, max_enc_step, max_dec_step, is_copy, single_pass = True , beam_search = True )
    return train_loader, valid_loader, test_loader

def get_loader(option):
    vocab = get_vocab(option)
    datas = get_datas(option, vocab)
    return (vocab, *datas)

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    vocab, train_loader, valid_loader, test_loader = get_loader(option)

    print(vocab.size) # 50000
    print(
        'word2id:\n',                                                                             # id2word
        vocab.special['UNK_TOKEN'], '\t', vocab.word2id.get(vocab.special['UNK_TOKEN']), '\n',    # [UNK]   0
        vocab.special['PAD_TOKEN'], '\t', vocab.word2id.get(vocab.special['PAD_TOKEN']), '\n',    # [PAD]   1
        vocab.special['BOS_TOKEN'], '\t', vocab.word2id.get(vocab.special['BOS_TOKEN']), '\n',    # [BOS]   2
        vocab.special['EOS_TOKEN'], '\t', vocab.word2id.get(vocab.special['EOS_TOKEN']), sep = '' # [EOS]   3
    )
    print(
        'id2word:\n',                            # id2word
        0, '\t', vocab.id2word.get(0), '\n',     # 0    [UNK]
        1, '\t', vocab.id2word.get(1), '\n',     # 1    [PAD]
        2, '\t', vocab.id2word.get(2), '\n',     # 2    [BOS]
        3, '\t', vocab.id2word.get(3), sep = ''  # 3    [EOS]
    )

    batch = train_loader.get_batch()
    if batch is not None:

        print(type(batch.src)) # nested list (batch_size, src_len)
        print(type(batch.trg)) # nested list (batch_size, trg_len)

        print(batch.enc_inp.shape)      # (batch_size, enc_step)
        print(batch.enc_len.shape)      # (batch_size,)
        print(batch.enc_pad_mask.shape) # (batch_size, enc_step)

        print(batch.dec_inp.shape)      # (batch_size, max_dec_step)
        print(batch.dec_out.shape)      # (batch_size, max_dec_step)
        print(batch.dec_len.shape)      # (batch_size,)
        print(batch.dec_pad_mask.shape) # (batch_size, max_dec_step)

        if  option.is_copy:
            print(batch.enc_inp_extend_vocab.shape) # (batch_size, enc_step)
            print(type(batch.oovs)) # nested list (batch_size, oov_num)
