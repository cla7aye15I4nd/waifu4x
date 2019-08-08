from PIL import Image

im = Image.open('../readonly/example/sample2.png')
im.resize((512, 512), Image.BICUBIC).save('sample2512.png')

# im = im.resize((1024, 1024), Image.BICUBIC)

# x = 500
# y = 50
# im.crop((x, y, x + 256, y + 256)).show()
