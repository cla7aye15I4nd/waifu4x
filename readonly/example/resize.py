from PIL import Image

Image.open('target.png').resize((256, 256), Image.ANTIALIAS).save('target.png')
