from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def loss_median(y_t, f):
    """
    Median Absolute Error. If q=0.5 the metric is Median Absolute Error.
    :param y_t: target value
    :param f: predicted value
    :return: Median Absolute Error
    """
    q = 0.50
    y_pred = ops.convert_to_tensor_v2_with_dispatch(f)
    y_true = math_ops.cast(y_t, y_pred.dtype)
    err = (y_true - y_pred)
    # return K.mean(K.maximum(q * err, (q - 1) * err), axis=-1)
    return K.mean(math_ops.maximum(q * err, (q - 1) * err), axis=-1)


def loss_max(y_true, y_predict):
    """
    Take the maximum of the MAE detectors.
    :param y_true: y target
    :param y_predict: y predicted by the NN
    :return: max_i(MAE_i)
    """
    # Define Loss as max_i(det_ran_error)
    loss_mae_none = losses.MeanAbsoluteError(reduction=losses.Reduction.NONE)
    a = tf.math.reduce_max(loss_mae_none(y_true, y_predict))  # axis=0
    return a
