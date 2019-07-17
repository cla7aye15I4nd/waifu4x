import os
import numpy as np
import tensorflow as tf
from PIL import Image

import config

class dataSet:
    def __init__(self, path, width = config.image_width, height = config.image_height):
        self.img_set = os.listdir(path)
        self.path = path
        self.width = width
        self.height = height
        self.gen_img = (img for img in self.img_set)

    def handler(self, img):
        im = Image.open(img).resize((self.width, self.height), Image.ANTIALIAS)
        return np.asarray(im) / 255
        
    def batch(self, batch_size = config.batch_size):
        return np.asarray([self.handler(os.path.join(self.path, next(self.gen_img))) for i in range(batch_size)])

def load_data():
    return (dataSet(config.data_train_LR, config.image_width, config.image_height),
            dataSet(config.data_train_HR, config.origin_width, config.origin_height),
            dataSet(config.data_valid_LR, config.image_width, config.image_height),
            dataSet(config.data_valid_HR, config.origin_width, config.origin_height))

if __name__ == '__main__':
    D = dataSet(config.data_train_HR)
    print(D.batch())
