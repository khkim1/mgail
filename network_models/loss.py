import tensorflow as tf


def l2_loss_masked(pred, tv):
    # Mask losses
    diff = (pred - tv[:, :-1]) * tf.expand_dims(tv[:, -1], axis=1)
    return tf.reduce_mean(tf.square(diff))

def l2_loss(pred, tv):
    # Mask losses
    diff = (pred - tv[:, :-1])
    return tf.reduce_mean(tf.square(diff))
