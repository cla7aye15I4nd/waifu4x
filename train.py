import json
import model
import config

import logging
logging.disable(logging.WARNING)

if __name__ == '__main__':
    with open(config.config_name, 'r') as f:
        json = json.loads(f.read())

    if 'global_step' in json:
        config.global_step = json['global_step']
            
    srgan = model.SRGAN(True)
    srgan.train()
    srgan.Sess.close()

    with open(config.config_name, 'w') as f:
        f.write(json.dumps({'global_step': config.global_step}))

    # TODO train output.json
