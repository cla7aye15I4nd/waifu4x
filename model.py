import tensorflow as tf
import time

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
        self.Sess =  tf.compat.v1.Session()
        self.G_inputs, self.G = Generator(config.G_input_shape, is_training)
        self.D_inputs, self.D = Discriminator(config.G_output_shape, is_training)

    def Gen(self, inputs):
        return self.Sess.run([self.G], feed_dict = {self.G_inputs : inputs})
        
    def trainG(self, X_train, y_train):
        for step in range(config.rounds):
            step_time = time.time()

            imageLR = X_train.batch()
            labelHR = y_train.batch()
        
            with tf.GradientTape() as tape:
                fakeHR = self.Gen(imageLR)

                print(labelHR.shape)
                print(fakeHR.get_shape())
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
