import tensorflow as tf
import numpy as np
import time
import os
import sys
import vgg19
from PIL import Image

import config
from data import load_data
from utils import (Input,
                   Conv2D,
                   PReLU,
                   LeakyReLU,
                   Sigmoid,
                   BatchNormal,
                   SubPixelConv2d,
                   Flatten,
                   Dense)


def Discriminator(inputs, reuse, is_training):
    def discriminator_block(inputs, output_channel, kernel_size, stride, is_training):
        net = Conv2D(inputs, kernel_size, output_channel, stride)
        net = BatchNormal(net, is_training)
        net = LeakyReLU(net, 0.2)
        return net

    with tf.variable_scope('discriminator', reuse=reuse):
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

        return net


def Generator(inputs, reuse, is_training):
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

    with tf.variable_scope('generator', reuse=reuse):

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

    def save_model(self, Saver, my_global_step, my_write_meta_data=False):
        Saver.save(self.Sess, os.path.join(config.checkpoint_dir, config.model_name),
                   global_step=my_global_step, write_meta_graph=my_write_meta_data)

    def load_model(self, Saver, global_step=0):
        try:
            if global_step > 0:
                model = os.path.join(config.checkpoint_dir, f'{config.model_name}-{global_step}')
                Saver.restore(self.Sess, model)
                print("load successfully")
                return True
        except:
            pass
        
        init = tf.compat.v1.global_variables_initializer()
        self.Sess.run(init)
        print("load failed")
        return False

    def threshold(self, img):
        row, col, dim = img.shape
        for i in range(row):
            for j in range(col):
                for k in range(dim):
                    if img[i][j][k] > 255:
                        img[i][j][k] = 255
                    if img[i][j][k] < 0:
                        img[i][j][k] = 0
        return img

    def evaluate(self, img_set, num):
        for img in img_set:
            image = Image.open(img).convert('RGB')
        
            arr_image = np.asarray(image) / 255 * 2 - 1
            test_input = tf.placeholder(dtype=tf.float32, shape=[1, image.width, image.height, config.dim])
            test_output = Generator(test_input, reuse=True, is_training=True)
            output = self.Sess.run(test_output, feed_dict={test_input: [arr_image]})[0]
            output = self.threshold((output + 1) / 2 * 255)

            output = Image.fromarray(output.astype('uint8'))
            if num > 0:
                output.save(os.path.join(config.train_path, 'output', f"{os.path.basename(img).split('.')[0]}_{num}.png"))
            else:
                output.save(os.path.join(config.predict_path, 'output', os.path.basename(img)))

    def train(self, mode = 0):
        
        lr_v = tf.Variable(config.lr_init)
        g_optimizer_init = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)
        g_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)
        d_optimizer = tf.train.AdamOptimizer(lr_v, beta1=config.beta1)

        G_inputs = tf.placeholder(dtype=tf.float32, shape=config.G_input_shape)
        G_label = tf.placeholder(dtype=tf.float32, shape=config.G_output_shape)
        G_fake = Generator(G_inputs, reuse=False, is_training=True)

        vgg_fake_model = vgg19.Vgg19('vgg19.npy')
        vgg_label_model = vgg19.Vgg19('vgg19.npy')
        vgg_fake = vgg_fake_model.build((G_fake + 1) / 2)
        vgg_label = vgg_label_model.build((G_label + 1) / 2)

        mse_loss = tf.losses.mean_squared_error(G_label, G_fake)
        tf.summary.scalar("mse_loss", mse_loss)
        perceptual_loss = tf.losses.mean_squared_error(vgg_label, vgg_fake) * 0.006
        tf.summary.scalar("perceptual_loss", perceptual_loss)
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'generator' in var.name]
        op_step = g_optimizer_init.minimize(mse_loss, var_list=g_vars)

        logits_fake = Discriminator(G_fake, reuse=False, is_training=True)
        logits_real = Discriminator(G_label, reuse=True, is_training=True)
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]

        d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real))
        d_loss1 = tf.reduce_mean(d_loss1)
        tf.summary.scalar("d_loss1", d_loss1)
        d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake))
        d_loss2 = tf.reduce_mean(d_loss2)
        tf.summary.scalar("d_loss2", d_loss2)
        d_loss = d_loss1 + d_loss2
        tf.summary.scalar("d_loss", d_loss)

        g_gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake))
        g_gan_loss = 1e-3 * tf.reduce_mean(g_gan_loss)
        tf.summary.scalar("g_gan_loss", g_gan_loss)
        g_loss = g_gan_loss + perceptual_loss
        tf.summary.scalar("g_loss", g_loss)

        op_step_d = d_optimizer.minimize(d_loss, var_list=d_vars)
        op_step_g = g_optimizer.minimize(g_loss, var_list=g_vars)

        with tf.variable_scope('summary'):
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(config.log_dir_train, self.Sess.graph)
            self.Sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=0.5)
                    
        start_time = time.time()
        if not self.load_model(saver, config.global_step):
            config.global_step = 1
            for epoch in range(config.epoch_num_init):
                X_train, y_train = load_data()
                for step in range(config.rounds_init):
                    step_time = time.time()

                    imageLR = X_train.batch()
                    labelHR = y_train.batch()

                    _, mse_loss_val = self.Sess.run([op_step, mse_loss], feed_dict={G_inputs: imageLR, G_label: labelHR})

                    print(f'step: [{step}/{config.rounds_init}] time: {time.time() - step_time}s, mse: {mse_loss_val} ')
                    if (step % 10 == 0):
                        write_summary = self.Sess.run(merged_summary, feed_dict={G_inputs: imageLR, G_label: labelHR})
                        train_writer.add_summary(write_summary, config.global_step)

                config.global_step += 1
                self.evaluate(config.test_image, config.global_step)
                if(config.global_step % 10 == 0):
                    self.save_model(saver, my_global_step=config.global_step)
        else:
            config.global_step += 1

        if mode:
            self.evaluate(config.test_image, -1)
        else:
            # global_vars = tf.global_variables()
            # is_not_initialized = self.Sess.run([tf.is_variable_initialized(var) for var in global_vars])
            # not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            for epoch in range(config.epoch_num):
                X_train, y_train = load_data()

                for step in range(config.rounds):
                    step_time = time.time()

                    imageLR = X_train.batch()
                    labelHR = y_train.batch()
                    _, d_loss_val, d_loss1_val, d_loss2_val= self.Sess.run([op_step_d, d_loss, d_loss1,
                                            d_loss2], feed_dict={G_inputs: imageLR, G_label : labelHR})
                    _, perceptual_loss_val, g_gan_loss_val = self.Sess.run([op_step_g, perceptual_loss, g_gan_loss],
                                                                    feed_dict={G_inputs: imageLR, G_label : labelHR})
                    # write_summary = self.Sess.run(merged_summary, feed_dict={G_inputs: imageLR, G_label : labelHR})
                    # train_writer.add_summary(write_summary, config.global_step)
                    g_loss_val = perceptual_loss_val + g_gan_loss_val

                    print(f'step: [{step}/{config.rounds}] step time: {time.time() - step_time}s total_time: {time.time() - start_time}')
                    print(f'        d_loss: {d_loss_val} d_loss1:{d_loss1_val} d_loss2:{d_loss2_val}')
                    print(f'        perceptual_loss:{perceptual_loss_val}, g_gan_loss:{g_gan_loss_val}, g_loss:{g_loss_val}')
                    if(step % 10 == 0):
                        write_summary = self.Sess.run(merged_summary, feed_dict={G_inputs: imageLR, G_label: labelHR})
                        train_writer.add_summary(write_summary, config.global_step)
                config.global_step += 1
                self.evaluate(config.test_image, config.global_step)
                self.save_model(saver, my_global_step=config.global_step)


if __name__ == '__main__':
    S = SRGAN(True)
