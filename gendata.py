import os
import sys
from PIL import Image

id = 0
path = r'E:\img'
hr_path = os.path.join('readonly', 'dataset', 'trainHR')
lr_path = os.path.join('readonly', 'dataset', 'trainLR')
algo = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.ANTIALIAS]

def search(path):
    global id, algo
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if os.path.isdir(f):
            search(f)
        else:
            try:
                hr = Image.open(f).convert('RGB')
                lr = hr.resize((hr.width // 4, hr.height // 4), algo[id % 4])
                hr.save(os.path.join(hr_path, f'{id}.png'))
                lr.save(os.path.join(lr_path, f'{id}.png'))
                id += 1
            except Exception as e:
                print(f'{e} : {f} Wrong')


if __name__ == '__main__':
    search(path)