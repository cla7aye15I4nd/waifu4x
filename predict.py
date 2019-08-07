import os
import json
import model
import config

import logging

logging.disable(logging.WARNING)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    with open(config.test_input, 'r') as f:
        js = json.loads(f.read())

    if 'input_path' in js:
        config.test_image = []
        for img in os.listdir(os.path.join(config.predict_path, js['input_path'])):
            config.test_image.append(os.path.join(config.predict_path, js['input_path'], img))
    if 'global_step' in js:
        config.global_step = js['global_step']

    srgan = model.SRGAN(True)
    srgan.train(mode=1)
    srgan.Sess.close()

    with open(config.test_output, 'w') as f:
        json.dump({'output_path': os.path.join(config.predict_path, 'output')}, f)
