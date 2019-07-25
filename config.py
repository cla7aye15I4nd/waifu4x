import os

#traing config
batch_size = 8
image_num = 1000
rounds_init = image_num // batch_size
rounds = image_num // batch_size
epoch_num = 100

#input config
dim = 3
ratio = 4
image_width = 24
image_height = 24
origin_width = image_width * ratio
origin_height = image_height * ratio


#data config
data_train_HR = os.path.join('readonly', 'dataset', 'trainHR')
data_train_LR = os.path.join('readonly', 'dataset', 'trainLR')
data_valid_HR = os.path.join('readonly', 'dataset', 'testHR')
data_valid_LR = os.path.join('readonly', 'dataset', 'testLR')

# network config
G_input_shape = [batch_size, image_width, image_height, dim]
G_output_shape = [batch_size, origin_width, origin_height, dim]
resblock_num = 16
subpixel_num = 2

#optimizer config
lr_init = 1e-4
beta1 = 0.9

#checkpoint
checkpoint_dir = os.path.join('longterm', 'checkpoint')
model_name = 'srgan-model'

#restore mode
global_step = 0

#config
test_input = os.path.join('temp', 'predict', 'input.json')
test_output = os.path.join('temp', 'predict', 'output.json')
train_output = os.path.join('temp', 'train', 'output.json')

config_name = 'config.json'
sample_image = os.path.join('readonly', 'example', 'sample.png')

test_image = [os.path.join('readonly', 'example', 'sample.png'),
              os.path.join('readonly', 'example', 'sample1.png'),
              os.path.join('readonly', 'example', 'sample2.png')]

train_path = os.path.join('temp', 'train')
predict_path = os.path.join('temp', 'predict')
