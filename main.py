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

logs_path = os.path.join(logs_folder, 'main.log')
save_path = os.path.join(save_folder, 'best.pth')

logger = get_logger(option.name, logs_path)

from loaders.loader1 import get_loader as get_loader1

from modules.module1 import get_module as get_module1

from utils.misc import train

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

logger.info('start training!')

interval = 100
for count_iter in range(option.total_iter):
    batch = train_loader.get_batch()
    loss, cove_loss = train(batch, seq2seq, device, optimizer, option.is_copy, option.is_coverage, option.cov_loss_wt, option.max_dec_step, option.grad_clip)
    if  count_iter % interval == 0:
        logger.info(
            'iter: %d, loss: %f, cover_loss: %f' %
            (count_iter, loss, cove_loss)
        )
