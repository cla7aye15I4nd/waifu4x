ratio = 4
batch_size = 128
image_width = 28
image_height = 28
dim = 3

G_input_shape = [batch_size, image_width, image_height, dim]
G_output_shape = [batch_size, image_width*ratio, image_height*ratio, dim]
resblock_num = 16
subpixel_num = 2
