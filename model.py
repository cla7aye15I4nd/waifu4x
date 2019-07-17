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


def Generator(input_shape, is_training):
    with tf.variable_scope('generator'):
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

        inputs = Input(input_shape)

        net = Conv2D(inputs, 9, 64, 1)
        net = PReLU(net)

        net = net + B_residual_block(net, 64, 1, is_training)

        for i in range(config.subpixel_num):
            net = SubPixelConv2d(net, 3, 256, 1)

        net = Conv2D(net, 9, 3, 1)

        return inputs, net

def Discriminator(input_shape, is_training):
    with tf.variable_scope('discriminator'):
        def discriminator_block(inputs, output_channel, kernel_size, stride, is_training):
            net = Conv2D(inputs, kernel_size, output_channel, stride)
            net = BatchNormal(net, is_training)
            net = LeakyReLU(net, 0.2)
            return net

        inputs = Input(input_shape)

        net = Conv2D(inputs, 3, 64, 1)
        net = LeakyReLU(net)

        net = discriminator_block(net,  64, 3, 2, is_training)
        net = discriminator_block(net, 128, 3, 1, is_training)
        net = discriminator_block(net, 128, 3, 2, is_training)
        net = discriminator_block(net, 256, 3, 1, is_training)
        net = discriminator_block(net, 256, 3, 2, is_training)
        net = discriminator_block(net, 512, 3, 1, is_training)
        net = discriminator_block(net, 512, 3, 2, is_training)

        net = Dense(Flatten(net), 1024)
        net = LeakyReLU(net, 0.2)

        net = Dense(net, 1)
        net = Sigmoid(net)

        return inputs, net
        

class SRGAN:
    def __init__(self, is_training):
        self.Sess = tf.compat.v1.Session()
        self.G_inputs, self.G = Generator(config.G_input_shape, is_training)
        self.D_inputs, self.D = Discriminator(config.G_output_shape, is_training)

#    def Gen(self, inputs):
#       return self.Sess.run([self.G], feed_dict={self.G_inputs: inputs})[0]

#    def Dis(self, inputs):
#        return self.Sess.run([self.D], feed_dict={self.D_inputs: inputs})[0]
        
    def trainG(self, X_train, y_train):
        lr_v = tf.Variable(config.lr_init)
        g_optimizer_init = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)  # .minimize(mse_loss, var_list=g_vars)
        g_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)  # .minimize(g_loss, var_list=g_vars)
        d_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)  # .minimize(d_loss, var_list=d_vars)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        G_label = tf.placeholder(dtype=tf.float32, shape=config.G_output_shape)
        fakeHR = self.G
        mse_loss = tf.losses.mean_squared_error(G_label, fakeHR)
        op_step = g_optimizer.minimize(mse_loss, var_list=g_vars)


        init = tf.compat.v1.global_variables_initializer()
        self.Sess.run(init)
        for step in range(config.rounds_init):
            step_time = time.time()

            my_imageLR = X_train.batch()
            my_labelHR = y_train.batch()

            step += 1
            _, mse_loss_print = self.Sess.run([op_step, mse_loss], feed_dict={self.G_inputs:my_imageLR, G_label:my_labelHR})
            print("step: [{}/{}] time: {}s, mse: {} ".format(
            step, config.rounds_init, time.time() - step_time, mse_loss_print))


        #对抗
        for step in range(config.rounds):
            step_time = time.time()

            imageLR = X_train.batch()
            labelHR = y_train.batch()

            with tf.GradientTape(persistent = True) as tape:
                fakeHR = self.Gen(imageLR)
                logits_fake = self.Dis(fakeHR)
                logits_real = self.Dis(labelHR)

                d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real))
                d_loss1 = tf.reduce_mean(d_loss1)
                d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake))
                d_loss2 = tf.reduce_mean(d_loss2)
                d_loss = d_loss1 + d_loss2

                g_gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake))
                g_gan_loss = 1e-3 * tf.reduce_mean(g_gan_loss)
                mse_loss = tf.losses.mean_squared_error(labelHR, fakeHR)
                g_loss = g_gan_loss + mse_loss

                grad = tape.gradient(d_loss, d_vars)
                op_step_t = d_optimizer.apply_gradients(zip(grad, d_vars))
                _, mse_loss_print, g_gan_loss_print = self.Sess.run([op_step_t, mse_loss, g_gan_loss])
                grad = tape.gradient(g_loss, g_vars)
                op_step_g = g_optimizer.apply_gradients(zip(grad, g_vars))
                _, d_loss_print = self.Sess.run([op_step_g, d_loss])

                step += 1
                print("step: [{}/{}] time: {}s, g_loss(mse:{},  g_gan_loss:{}) d_loss: {}".format(
                    step, config.rounds, time.time() - step_time, mse_loss_print, g_gan_loss_print, d_loss_print))








                #print(labelHR.shape)
                #print(fakeHR.get_shape())
                #mse_loss = tf.losses.mean_squared_error(labelHR, fakeHR)

                #grad = tape.gradient(mse_loss, G.weights)
                # g_optimizer_init.apply_gradients(zip(grad, G.weights))
                # step += 1
                # epoch = step//n_step_epoch
                # print("Epoch: [{}/{}] step: [{}/{}] time: {}s, mse: {} ".format(
                #     epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
                # if (epoch != 0) and (epoch % 10 == 0):
                #    tl.vis.save_images(fake_hr_patchs.numpy(), [ni, ni], save_dir_gan + '/train_g_init_{}.png'.format(epoch))


                    
if __name__ == '__main__':
    S = SRGAN(True)
