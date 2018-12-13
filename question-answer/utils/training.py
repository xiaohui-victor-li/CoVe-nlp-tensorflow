# helper function to facilitate model training

import tensorflow as tf
import numpy as np

from timeit import default_timer as timer
from collections import Counter


_PAD = "<pad>"
_SOS = "<sos>"
_UNK = "<unk>"

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

def f1(prediction, truth):
    total = 0
    f1_total = 0
    for i, single_truth in enumerate(truth):
        total += 1
        single_prediction = prediction[0][i], prediction[1][i]
        f1 = f1_score(single_prediction, single_truth)
        f1_total += f1
    f1_total /= total

    return f1_total

def f1_score(prediction, truth):
    start, end = truth
    true_range = range(start, end+1)
    start_pred, end_pred = prediction
    prediction_range = range(start_pred, end_pred+1)
    common = Counter(prediction_range) & Counter(true_range)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_range)
    recall = 1.0 * num_same / len(true_range)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match(prediction, truth):
    total = 0
    em_total = 0
    for i, single_truth in enumerate(truth):
        if [prediction[0][i], prediction[1][i]] == single_truth:  # can possibly remove loop and just do the full comparison
            em_total +=1
        total += 1
    em_total /= total
    return em_total

def reverse_indices(indices, rev_vocab):
    """ Recovers words from embedding indices

    Args:
        indices: Integer indices to recover words for.
        rev_vocab: Reverse vocabulary. Dictionary mapping indices to words.

    Returns:
        String of words with space as separation
    """
    return ' '.join([rev_vocab[idx] for idx in indices if idx != PAD_ID])

def multibatch_prediction_truth(session, model, data, FLAGS, num_batches=None, random=False):
    """ Returns batches of predictions and ground truth answers.

    Args:
        session: TensorFlow Session.
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders.
        question, question_length, paragraph, paragraph_length.
        data: SquadDataset data to do minibatch evaluation on.
        num_batches: Number of batches of size FLAGS.batch_size to evaluate over. `None` for whole data set.
        random: True for random and possibly overlapping batches. False for deterministic sequential non-overlapping batches.

    Returns:
        Tuple of
            Predictions, tuple of two numpy arrays containing start and end of answer spans
            Truth, list of tuples containing start and end of answer spans
    """
    if num_batches is None:
        num_batches = data.length // FLAGS.batch_size
    truth = []
    start = []
    end = []
    for i in range(num_batches):
        if random:
            q, p, ql, pl, a = data.get_batch(FLAGS.batch_size)
        else:
            begin_idx = i * FLAGS.batch_size
            q, p, ql, pl, a = data[begin_idx:begin_idx + FLAGS.batch_size]
        answer_start, answer_end = session.run(model.answer, model.fill_feed_dict(q, p, ql, pl))
        # for i, s in enumerate(answer_start):
        #     if s > answer_end[i]:
        #         print('predicted: ', (s, answer_end[i], pl[i]), 'truth: ', (a[i][0], a[i][1]))
        start.append(answer_start)
        end.append(answer_end)
        truth.extend(a)
    start = np.concatenate(start)
    end = np.concatenate(end)
    prediction = (start, end)
    return prediction, truth


def test_overfit(model, train, FLAGS):
    """ Tests that model can overfit on small datasets.

    Args:
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders
        question, question_length, paragraph, paragraph_length
        train: Training set
    """
    epochs = 100
    test_size = 32
    steps_per_epoch = 10
    train.question, train.paragraph, train.question_length, train.paragraph_length, train.answer = train[:test_size]
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_start = timer()
            for step in range(steps_per_epoch):
                feed_dict = model.fill_feed_dict(*train[:test_size], is_training=True)
                fetch_dict = {
                    'step': tf.train.get_global_step(),
                    'loss': model.loss,
                    'train': model.train
                }
                result = session.run(fetch_dict, feed_dict)
                loss = result['loss']

                if (step == 0 and epoch == 0):
                    print(f'Entropy - Result: {loss:.2f}, Expected (approx.): {2*np.log(FLAGS.max_paragraph_length):.2f}')
                if step == steps_per_epoch-1:
                    print(f'Cross entropy: {loss:.2f}')
                    train.length = test_size
                    prediction, truth = multibatch_prediction_truth(session, model, train, 1)
                    overfit_f1 = f1(prediction, truth)
                    print(f'F1: {overfit_f1:.2f}')
            global_step = tf.train.get_global_step().eval()
            print(f'Epoch took {timer() - epoch_start:.2f} s (step: {global_step})')