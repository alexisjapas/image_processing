import cv2
import matplotlib.pyplot as plt

import libs.ip_utils as ip


matrix = cv2.imread('data/kolibri.jpg')

pixel = ip.pixelized_matrix(matrix, 400, 40)
plt.figure(1)
plt.imshow(pixel, cmap='gray')
plt.show()
