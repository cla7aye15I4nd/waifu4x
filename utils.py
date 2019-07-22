import tensorflow as tf
import config

from tensorflow.compat.v1 import placeholder, Variable, get_variable
from tensorflow.contrib import slim

def Input(input_shape):
    return placeholder(tf.float32, input_shape)

def Conv2D(inputs, kernel=3, output_channel=64, stride=1):
    return slim.conv2d(inputs, output_channel, [kernel, kernel],
                       stride, 'SAME', data_format='NHWC', activation_fn=None,
                       weights_initializer=tf.contrib.layers.xavier_initializer())
    
def PReLU(inputs, alpha=0.2):
    pos = tf.nn.relu(inputs)
    neg = alpha * (inputs - abs(inputs)) * 0.5

    return pos + neg

def LeakyReLU(inputs, alpha=0.2):
    return tf.nn.leaky_relu(inputs, alpha)

def Sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def BatchNormal(inputs, is_training=True):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001,
                           updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
                           scale=False, fused=True, is_training=is_training)

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

def PixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def SubPixelConv2d(inputs, kernel=3, output_channel=256, stride=1):
    net = Conv2D(inputs, kernel, output_channel, stride)
    net = PixelShuffler(net, scale=2)
    net = PReLU(net)
    return net

def Flatten(inputs):
    return slim.flatten(inputs)

def Dense(inputs, output_size):
    return tf.layers.dense(inputs, output_size, activation=None,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())
