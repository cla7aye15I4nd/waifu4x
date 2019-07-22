import tensorflow as tf
import numpy as np
import time
import sys

import config
from utils import (Input,
                   Conv2D,
                   PReLU,
                   LeakyReLU,
                   Sigmoid,
                   BatchNormal,
                   SubPixelConv2d,
                   Flatten,
                   Dense)

def Discriminator(inputs, my_reuse, is_training):
    def discriminator_block(inputs, output_channel, kernel_size, stride, is_training):
        net = Conv2D(inputs, kernel_size, output_channel, stride)
        net = BatchNormal(net, is_training)
        net = LeakyReLU(net, 0.2)
        return net

    with tf.variable_scope('discriminator', reuse=my_reuse):
        net = Conv2D(inputs, 3, 64, 1)
        net = LeakyReLU(net)

        net = discriminator_block(net, 64, 3, 2, is_training)
        net = discriminator_block(net, 128, 3, 1, is_training)
        net = discriminator_block(net, 128, 3, 2, is_training)
        net = discriminator_block(net, 256, 3, 1, is_training)
        net = discriminator_block(net, 256, 3, 2, is_training)
        net = discriminator_block(net, 512, 3, 1, is_training)
        net = discriminator_block(net, 512, 3, 2, is_training)

        net = Dense(Flatten(net), 1024)
        net = LeakyReLU(net, 0.2)

        net = Dense(net, 1)
        #net = Sigmoid(net)

        return net

def Generator(inputs, my_reuse, is_training):
    def residual_blocks(inputs, output_channel, stride):
        net = Conv2D(inputs, 3, output_channel, stride)
        net = BatchNormal(net)
        net = PReLU(net)
        net = Conv2D(net, 3, output_channel, stride)
        return net + inputs

    def B_residual_block(net, output_channel, stride, is_training):
        for i in range(config.resblock_num):
            net = residual_blocks(net, output_channel, stride)

        net = Conv2D(net, 3, 64, 1)
        net = BatchNormal(net, is_training)
        return net

    with tf.variable_scope('generator', reuse=my_reuse):

        net = Conv2D(inputs, 9, 64, 1)
        net = PReLU(net)

        net = net + B_residual_block(net, 64, 1, is_training)

        for i in range(config.subpixel_num):
            net = SubPixelConv2d(net, 3, 256, 1)

        net = Conv2D(net, 9, 3, 1)

        return net

class SRGAN:
    def __init__(self, is_training):
        self.Sess = tf.compat.v1.Session()

    def save(self, step):
        self.Saver.save(self.Sess, os.path.join(config.checkpoint_dir, config.model_name))

    def restore(self):
        fn = os.path.join(config.checkpoint_dir, config.model_name)
        if os.path.exist(fn):
            self.Saver.restore(self.Sess, fn)

    def train(self, X_train, y_train):
        lr_v = tf.Variable(config.lr_init)
        g_optimizer_init = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)
        g_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)
        d_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)

        G_inputs = tf.placeholder(dtype=tf.float32, shape=config.G_input_shape)
        G_label = tf.placeholder(dtype=tf.float32, shape=config.G_output_shape)
        G_fake = Generator(G_inputs, my_reuse=False, is_training=True)
        # 初始训练generator操作

        mse_loss = tf.losses.mean_squared_error(G_label, G_fake)
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'generator' in var.name]
        op_step = g_optimizer_init.minimize(mse_loss, var_list=g_vars)
        # 对抗训练操作

        logits_fake = Discriminator(G_fake, my_reuse=False, is_training=True)
        logits_real = Discriminator(G_label, my_reuse=True, is_training=True)
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]

        d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
        d_loss1 = tf.reduce_mean(d_loss1)
        d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        d_loss2 = tf.reduce_mean(d_loss2)
        d_loss = d_loss1 + d_loss2
        # d_loss = tf.add(d_loss1, d_loss2)

        g_gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
        g_gan_loss = 1e-3 * tf.reduce_mean(g_gan_loss)
        g_loss = g_gan_loss + mse_loss

        op_step_d = d_optimizer.minimize(d_loss, var_list=d_vars)
        op_step_g = g_optimizer.minimize(g_loss, var_list=g_vars)
        # 开始训练

        init = tf.compat.v1.global_variables_initializer()
        self.Sess.run(init)

        for step in range(config.rounds_init):
            step_time = time.time()

            imageLR = X_train.batch()
            labelHR = y_train.batch()

            _, mse_loss_print = self.Sess.run([op_step, mse_loss], feed_dict={G_inputs: imageLR, G_label: labelHR})

            # mse_loss_print = self.Sess.run([mse_loss],
            #                                   feed_dict={self.G_inputs: my_imageLR, G_label: my_labelHR})
            # _ = self.Sess.run([op_step], feed_dict={self.G_inputs: imageLR, G_label: labelHR})

            print("step: [{}/{}] time: {}s, mse: {} ".format(
                step, config.rounds_init, time.time() - step_time, mse_loss_print))

        # 对抗
        for step in range(config.rounds):
            step_time = time.time()

            imageLR = X_train.batch()
            labelHR = y_train.batch()
            _, d_loss_print, d_loss1_print, d_loss2_print = self.Sess.run([op_step_d, d_loss, d_loss1, d_loss2],
                                                                          feed_dict={G_inputs: imageLR,
                                                                                     G_label: labelHR})
            _, mse_loss_print, g_gan_loss_print = self.Sess.run([op_step_g, mse_loss, g_gan_loss],
                                                                feed_dict={G_inputs: imageLR, G_label: labelHR})
            g_loss_print = mse_loss_print + g_gan_loss_print
            # print("step: [{}/{}] time: {}s, mse_loss(mse:{},g_gan_loss:{})   d_loss: {} d_loss1:{} d_loss2:{}".format(
            #     step, config.rounds, time.time() - step_time, mse_loss_print, g_gan_loss_print, d_loss_print, d_loss1_print, d_loss2_print))

            print("step: [{}/{}] time: {}s, d_loss: {} d_loss1:{} d_loss2:{}".format(step, config.rounds, time.time()
                                                                                     - step_time, d_loss_print,
                                                                                     d_loss1_print, d_loss2_print))
            print(
                "step: [{}/{}] time: {}s, mse_loss:{}, g_gan_loss:{}, g_loss:{}".format(step, config.rounds, time.time()
                                                                                        - step_time, mse_loss_print,
                                                                                        g_gan_loss_print, g_loss_print))

        self.save()


if __name__ == '__main__':
    S = SRGAN(True)
