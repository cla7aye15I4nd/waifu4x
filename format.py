import os
import sys
from PIL import Image

img_set = os.listdir(os.path.join('anime', 'train'))

id = 0
for img in img_set:
    try:
        HR = Image.open(os.path.join('anime', 'train', img)).convert('RGB')
        LR = HR.resize((HR.width // 4, HR.height // 4), Image.ANTIALIAS)
        HR.save(os.path.join('anime', 'trainHR', f'{id}.png'))
        LR.save(os.path.join('anime', 'trainLR', f'{id}.png'))
        id += 1
    except:
        pass

id = 0
img_set = os.listdir(os.path.join('anime', 'test'))
for img in img_set:
    try:
        HR = Image.open(os.path.join('anime', 'test', img)).convert('RGB')
        LR = HR.resize((HR.width // 4, HR.height // 4), Image.ANTIALIAS)
        HR.save(os.path.join('anime', 'testHR', f'{id}.png'))
        LR.save(os.path.join('anime', 'testLR', f'{id}.png'))
        id += 1
    except:
        pass