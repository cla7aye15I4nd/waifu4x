import time
import tensorflow as tf

from data import load_data
import model
import config

import logging
logging.disable(logging.WARNING)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    SRGAN = model.SRGAN(True)

    #init = tf.compat.v1.global_variables_initializer()
    #SRGAN.Sess.run(init)
    
    #init G
    SRGAN.trainG(X_train, y_train) 
    
    SRGAN.Sess.close()
