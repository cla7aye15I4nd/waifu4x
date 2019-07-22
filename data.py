import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import cm
from random import randint
import config


class dataSet:
    def __init__(self, seed, tag, path, width=config.image_width, height=config.image_height):
        self.seed = seed
        self.tag = tag
        self.img_set = os.listdir(path)
        self.path = path
        self.width = width
        self.height = height
        self.gen_img = (img for img in self.img_set)
        self.pos = 0

    def show(self, rgb):
        im = Image.fromarray((rgb * 255).astype('uint8'))
        im.show()

    def handler(self, img):
        image = Image.open(img)
        x, y = self.seed[self.pos]
        x = x % (image.width - self.width)
        y = y % (image.height - self.height)
        if self.tag:
            x *= config.ratio
            y *= config.ratio
        self.pos += 1
        im = np.asarray(image.crop((x, y, x + self.width, y + self.height))) / 255
        return im
        #self.show(im)

    def batch(self, batch_size=config.batch_size):
        return np.asarray([self.handler(os.path.join(self.path, next(self.gen_img)))
                           for i in range(batch_size)])


def load_data():
    seed = [(randint(0, 127), randint(0, 127)) for x in range(config.image_num)]
    return (dataSet(seed, 0, config.data_train_LR, config.image_width, config.image_height),
            dataSet(seed, 1, config.data_train_HR, config.origin_width, config.origin_height),
            dataSet(seed, 0, config.data_valid_LR, config.image_width, config.image_height),
            dataSet(seed, 1, config.data_valid_HR, config.origin_width, config.origin_height))


if __name__ == '__main__':
    D = dataSet(config.data_train_HR)
    print(D.batch())