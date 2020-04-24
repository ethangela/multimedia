# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pprint


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--video_root_dir', type=str, default='.') 
    parser.add_argument('--save_dir', type=str, default='./checkpoint_new_lstm_april_23/') 
    parser.add_argument('--score_dir', type=str, default='./evaluate_result.pkl') 
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/_epoch-49.pkl')

    # Model
    parser.add_argument('--attention_mode', type=str2bool, default='true')
    parser.add_argument('--decoder_mode', type=str, default='LSTM')
    parser.add_argument('--v_input_size', type=int, default=2048)
    parser.add_argument('--v_hidden_size', type=int, default=512)
    parser.add_argument('--q_input_size', type=int, default=768)
    parser.add_argument('--q_hidden_size', type=int, default=192)
    parser.add_argument('--lstm_input_size', type=int, default=704)
    parser.add_argument('--lstm_hidden_size', type=int, default=352) 
    parser.add_argument('--mlp_input_size', type=int, default=704) 
    parser.add_argument('--mlp_hidden_size1', type=int, default=512)
    parser.add_argument('--mlp_hidden_size2', type=int, default=128)

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--clip', type=float, default=5.0) ###
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()

