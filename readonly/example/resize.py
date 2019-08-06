from PIL import Image

Image.open('sample.png').resize((1024, 1024), Image.BICUBIC).save('target.png')
