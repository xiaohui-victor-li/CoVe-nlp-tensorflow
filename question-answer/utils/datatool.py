# helper functions for data process

import os
from os.path import join as pjoin
import tensorflow as tf


def initialize_vocab(vocab_path):
    # CS224n
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def get_normalized_train_dir(train_dir):
    # CS224n
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def get_data_paths(data_dir, name='train'):
    question_file = pjoin(data_dir, f'{name}.ids.question')
    paragraph_file = pjoin(data_dir, f'{name}.ids.context')
    answer_file = pjoin(data_dir, f'{name}.span')
    return question_file, paragraph_file, answer_file




