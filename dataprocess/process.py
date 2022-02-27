import os
import subprocess

import glob
import struct
import pandas as pd

from tensorflow.core.example import example_pb2

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

subprocess.run('mkdir -p %s' % option.sources_path, shell = True)
subprocess.run('mkdir -p %s' % option.targets_path, shell = True)

if not os.path.exists (
    os.path.join(option.sources_path, 'finished_files')
):
    raise FileNotFoundError (
        'Please download dataset files into %s folder' % option.sources_path
    )

source_vocab_path = os.path.join(option.sources_path, 'finished_files/vocab')

source_train_path = os.path.join(option.sources_path, 'finished_files/chunked/train_*')
source_valid_path = os.path.join(option.sources_path, 'finished_files/chunked/val_*')
source_test_path  = os.path.join(option.sources_path, 'finished_files/chunked/test_*')

target_vocab_path = os.path.join(option.targets_path, option.vocab_file)

target_train_path = os.path.join(option.targets_path, option.train_file)
target_valid_path = os.path.join(option.targets_path, option.valid_file)
target_test_path  = os.path.join(option.targets_path, option.test_file )

vocab = []

with open(source_vocab_path, 'r', encoding = 'utf-8') as txt_file:
    for line in txt_file:
        item = line.split()
        if len(item) != 2:
            continue
        vocab.append(item)

with open(target_vocab_path, 'w', encoding = 'utf-8') as txt_file:
    for word, freq in vocab:
        txt_file.write(word + '\t' + freq + '\n')

def load_datas(file_path):
    datas = []
    files = glob.glob(file_path)
    for file_obj in files:
        with open(file_obj, 'rb') as bin_file:
            while True:
                meta_len = 8
                meta_buf = bin_file.read(meta_len)
                if not meta_buf: break
                data_len = struct.unpack('q', meta_buf)[0]
                data_buf = bin_file.read(data_len)
                data_str = struct.unpack('%ds' % data_len, data_buf)[0]
                data  = example_pb2.Example.FromString(data_str)

                try:
                    article  = data.features.feature['article'] .bytes_list.value[0]
                    abstract = data.features.feature['abstract'].bytes_list.value[0]
                except:
                    continue

                if (
                    len(article) == 0 or
                    len(abstract) == 0
                ):
                    continue
                else:
                    abstract_bos = '<s>' .encode()
                    abstract_eos = '</s>'.encode()
                    abstract_bos_len = len(abstract_bos)
                    abstract_eos_len = len(abstract_eos)
                    abstract_splits = []
                    cur_p = 0
                    while True:
                        try:
                            sta_p = abstract.index(abstract_bos, cur_p)
                            end_p = abstract.index(abstract_eos, sta_p + 1)
                            cur_p = end_p + abstract_eos_len
                            cus_s = abstract[sta_p + abstract_bos_len: end_p]
                            abstract_splits.append(cus_s.strip())
                        except:
                            break

                    final_article  = article.decode()
                    final_abstract = ' '.encode().join(abstract_splits).decode()

                    datas.append({
                        'article' : final_article,
                        'abstract': final_abstract
                    })

    return datas

train_datas = load_datas(source_train_path)
valid_datas = load_datas(source_valid_path)
test_datas  = load_datas(source_test_path )

pd.DataFrame(train_datas).to_csv(target_train_path, sep = '\t', index = None)
pd.DataFrame(valid_datas).to_csv(target_valid_path, sep = '\t', index = None)
pd.DataFrame(test_datas ).to_csv(target_test_path , sep = '\t', index = None)
