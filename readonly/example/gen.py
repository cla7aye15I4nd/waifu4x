import os
import sys
from PIL import Image

algo = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.ANTIALIAS]

im = Image.open('trans.png').convert('RGB')

im.resize((im.width // 4, im.height // 4), algo[0]).save('trans_0.png')
im.resize((im.width // 4, im.height // 4), algo[1]).save('trans_1.png')
#im.resize((im.width // 4, im.height // 4), algo[2]).save('trans_2.png')
#im.resize((im.width // 4, im.height // 4), algo[3]).save('trans_3.png')
im.resize((im.width // 4, im.height // 4)).save('trans_2.png')
