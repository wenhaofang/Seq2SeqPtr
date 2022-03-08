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
    parser.add_argument('--emb_dim', type = int, default = 256, help = '')
    parser.add_argument('--hid_dim', type = int, default = 512, help = '')
    parser.add_argument('--dropout', type = float, default = 0.5, help = '')

    parser.add_argument('--is_copy', action = 'store_true', help = '')
    parser.add_argument('--is_coverage', action = 'store_true', help = '')
    parser.add_argument('--cov_loss_wt', type = float, default = 1.0, help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 16, help = '')
    parser.add_argument('--total_iter', type = int, default = 500000, help = '')

    parser.add_argument('--lr', type = float, default = 0.15, help = '')
    parser.add_argument('--lr_coverage', type = float, default = 0.15, help = '')
    parser.add_argument('--ada_init_ac', type = float, default = 0.1, help = '')

    parser.add_argument('--grad_clip', type = float, default = 2.0, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
