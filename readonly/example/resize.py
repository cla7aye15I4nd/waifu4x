from PIL import Image

Image.open('panda.png').resize((212*4, 238*4), Image.BICUBIC).save('panda_bic.png')
