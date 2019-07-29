import os
import json
import model
import config

import logging

logging.disable(logging.WARNING)

if __name__ == '__main__':
    with open(config.test_input, 'r') as f:
        json = json.loads(f.read())

    if 'input_path' in json:
        config.test_image = []
        for img in os.listdir(os.path.join(config.predict_path, json['input_path'])):
            config.test_image.append(os.path.join(config.predict_path, json['input_path'], img))
    if 'global_step' in json:
        config.global_step = json['global_step']

    srgan = model.SRGAN(True)
    srgan.train(mode=1)
    srgan.Sess.close()

    # TODO predict output.json
