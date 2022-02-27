import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')

    parser.add_argument('--vocab_file', default = 'vocab.tsv', help = '')
    parser.add_argument('--train_file', default = 'train_data.tsv', help = '')
    parser.add_argument('--valid_file', default = 'valid_data.tsv', help = '')
    parser.add_argument('--test_file' , default = 'test_data.tsv' , help = '')

    parser.add_argument('--min_freq', type = int, default = 6, help = '')
    parser.add_argument('--max_numb', type = int, default = 50000, help = '')
    parser.add_argument('--max_enc_step', type = int, default = 400, help = '')
    parser.add_argument('--max_dec_step', type = int, default = 100, help = '')

    # For Module
    parser.add_argument('--is_copy', action = 'store_true', help = '')
    parser.add_argument('--is_coverage', action = 'store_true', help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--iter_count', type = int, default = 500000, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
