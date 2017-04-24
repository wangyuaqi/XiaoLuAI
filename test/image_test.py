import cv2
import numpy as np

image = cv2.imread('/home/lucasx/Documents/crop_images/training_set/2016301110085.jpg')
b, g, r = cv2.split(image)
rgb = np.concatenate((r.reshape((144 * 144)), g.reshape((144 * 144)), b.reshape((144 * 144))), axis=0).reshape(
    1, 144 * 144 * 3)
print(rgb)
print(rgb.shape)
