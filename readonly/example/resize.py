from PIL import Image

Image.open('sample2HR.png').resize((256, 256), Image.ANTIALIAS).save('sample2.png')
