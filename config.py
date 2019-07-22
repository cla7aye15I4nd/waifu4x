import os

#traing config
batch_size = 8
image_num = 800
rounds = image_num // batch_size - 1
rounds_init = rounds // 10

#input config
ratio = 4
image_width = 96
image_height = 96
origin_width = image_width*ratio
origin_height = image_height*ratio
dim = 3

#data config
data_train_HR = os.path.join('data', 'DIV2K_train_HR')
data_train_LR = os.path.join('data', 'DIV2K_train_LR_bicubic', 'X4')
data_valid_HR = os.path.join('data', 'DIV2K_valid_HR')
data_valid_LR = os.path.join('data', 'DIV2K_valid_LR_bicubic', 'X4')

# network config
G_input_shape = [batch_size, image_width, image_height, dim]
G_output_shape = [batch_size, image_width*ratio, image_height*ratio, dim]
resblock_num = 16
subpixel_num = 2

#optimizer config
lr_init = 1e-4
beta1 = 0.9

#checkpoint
checkpoint_dir = os.path.join('data', 'checkpoint')
model_name = 'srgan.model'
