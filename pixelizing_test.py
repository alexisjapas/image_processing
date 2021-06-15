import cv2
import matplotlib.pyplot as plt

import libs.ip_utils as ip


matrix = cv2.imread('data/lenna.png')
plt.imshow(matrix, cmap='gray')
plt.show()

pixel = ip.pixelized_matrix(matrix, 3)
plt.imshow(pixel, cmap='gray')
plt.show()

pixel = ip.pixelized_matrix(matrix, 6)
plt.imshow(pixel, cmap='gray')
plt.show()

pixel = ip.pixelized_matrix(matrix, 9)
plt.imshow(pixel, cmap='gray')
plt.show()

pixel = ip.pixelized_matrix(matrix, 12)
plt.imshow(pixel, cmap='gray')
plt.show()

pixel = ip.pixelized_matrix(matrix, 30)
plt.imshow(pixel, cmap='gray')
plt.show()