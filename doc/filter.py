import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("image.jpg")
pylab.show()

ker = np.array([[ -1, 0, 1], 
                [ -2, 0, 2],
                [ -1, 0, 1]])

res = cv2.filter2D(img, -1, ker)

plt.imshow(res)            
plt.imsave("result.png",res)
