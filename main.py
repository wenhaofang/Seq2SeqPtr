import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)
result_folder = os.path.join(root_path, 'result', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logger = get_logger(option.name, os.path.join(logs_folder, 'main.log'))

from loaders.loader1 import get_loader as get_loader1

from modules.module1 import get_module as get_module1

from utils.misc import train, valid, test, save_checkpoint, load_checkpoint, save_sample, calc_matrix

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

vocab, train_loader, valid_loader, test_loader = get_loader1(option)

logger.info('prepare module')

seq2seq = get_module1(option, vocab.size)

seq2seq = seq2seq.to (device)

logger.info('prepare envs')

params_list = list(seq2seq.parameters())
ada_init_lr = option.lr_coverage if option.is_coverage else option.lr
ada_init_ac = option.ada_init_ac

optimizer = optim.Adagrad(params_list, lr = ada_init_lr, initial_accumulator_value = ada_init_ac)

print_interval = 100
check_interval = 1000

assert option.mode in ['train', 'valid', 'test']

if  option.mode == 'train':
    logger.info('start training!')

    if  option.ckpt != '' and os.path.isfile(option.ckpt):
        start_iter = load_checkpoint(option.ckpt, seq2seq, optimizer)
        logger.info('training a old model: %s' % (option.ckpt))
    else:
        start_iter = 0
        logger.info('training a new model')

    for count_iter in range(start_iter + 1, option.total_iter):
        batch = train_loader.get_batch()
        if  batch is None:
            break
        loss, cove_loss = train(batch, seq2seq, device, optimizer, option.is_copy, option.is_coverage, option.cov_loss_wt, option.max_dec_step, option.grad_clip)
        if  count_iter % print_interval == 0:
            logger.info('iter: %d, loss: %f, cove_loss: %f' % (count_iter, loss, cove_loss))
        if  count_iter % check_interval == 0:
            save_checkpoint(os.path.join(save_folder, '%s.ckpt' % str(count_iter)), seq2seq, optimizer, count_iter)

if  option.mode == 'valid':
    logger.info('start validing!')

    if  option.ckpt != '' and os.path.isfile(option.ckpt):
        final_iter = load_checkpoint(option.ckpt, seq2seq, optimizer)
        logger.info('validing a old model: %s' % (option.ckpt))
    else:
        logger.info('validing a new model, unexpected')
        raise Exception('Expect to use a pre-existing model')

    count_iter = 0
    while True:
        count_iter += 1
        batch = valid_loader.get_batch()
        if  batch is None:
            break
        loss, cove_loss = valid(batch, seq2seq, device, option.is_copy, option.is_coverage, option.cov_loss_wt, option.max_dec_step)
        if  count_iter % print_interval == 0:
            logger.info('iter: %d, loss: %f, cove_loss: %f' % (count_iter, loss, cove_loss))

if  option.mode == 'test':
    logger.info('start testing!')

    if  option.ckpt != '' and os.path.isfile(option.ckpt):
        final_iter = load_checkpoint(option.ckpt, seq2seq, optimizer)
        logger.info('testing a old model: %s' % (option.ckpt))
    else:
        logger.info('testing a new model, unexpected')
        raise Exception('Expect to use a pre-existing model')

    sources = []
    targets = []
    predict = []
    count_iter = 0
    while True:
        count_iter += 1
        batch = test_loader.get_batch()
        if  batch is None:
            break
        src, trg, pred = test(batch, seq2seq, device, option.is_copy, option.is_coverage, vocab, option.min_dec_step, option.max_dec_step, option.beam_width)
        sources.append(src)
        targets.append(trg)
        predict.append(pred)
        if  count_iter % print_interval == 0:
            logger.info('iter: %d\nsrc: %s\ntrg: %s\npred: %s' % (count_iter, src, trg, pred))

    save_sample(result_folder, final_iter, sources, targets, predict)
    calc_matrix(result_folder, final_iter, sources, targets, predict)
