# helper functions for network usage

import tensorflow as tf


#-------------------------------------------DynamicCoattentionNetwork-------------------------------------------

def maybe_mask_affinity(affinity, sequence_length, affinity_mask_value=float('-inf')):
    """ Masks affinity along its third dimension with `affinity_mask_value`.

    Used for masking entries of sequences longer than `sequence_length` prior to
    applying softmax.
    Args:
        affinity: Tensor of rank 3, shape [N, D or Q, Q or D] where attention logits are in the second dimension.
        sequence_length: Tensor of rank 1, shape [N]. Lengths of second dimension of the affinity.
        affinity_mask_value: (optional) Value to mask affinity with.
    Returns:
        Masked affinity, same shape as affinity.
    """
    if sequence_length is None:
        return affinity
    score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(affinity)[2]))
    affinity_mask_values = affinity_mask_value * tf.ones_like(affinity)
    return tf.where(score_mask, affinity, affinity_mask_values)

def maybe_mask_to_start(score, start, score_mask_value):
    score_mask = tf.sequence_mask(start, maxlen=tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(~score_mask, score, score_mask_values)

def maybe_dropout(keep_prob, is_training):
    return tf.cond(tf.convert_to_tensor(is_training), lambda: keep_prob, lambda: 1.0)


def start_and_end_encoding(encoding, answer):
    """ Gathers the encodings representing the start and end of the answer span passed
    and concatenates the encodings.

    Args:
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.
        answer: Tensor of rank 2. Answer span.

    Returns:
        Tensor of rank 2 [N, 2xH], containing the encodings of the start and end of the answer span
    """
    batch_size = tf.shape(encoding)[0]
    start, end = answer[:, 0], answer[:, 1]
    encoding_start = tf.gather_nd(encoding,
                                  tf.stack([tf.range(batch_size), start], axis = 1))  # May be causing UserWarning
    encoding_end = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), end], axis = 1))
    return tf.concat([encoding_start, encoding_end], axis = 1)