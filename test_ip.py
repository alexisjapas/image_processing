#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:04:07 2021

@author: pouja
"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import libs.ip_utils as ip


# Colored matrix
matrix = cv.imread("data/kolibri.jpg")

# Let's move to gray scale
# matrix_gray = ip.color_to_gray(matrix)
# print("Gray image:", matrix_gray.shape)
# plt.imshow(matrix_gray, cmap='gray')
# plt.show()
matrix_gray = matrix

# Calculate it's histogram
h = ip.histogram(matrix_gray, 8)
print(np.amax(h))
plt.plot(h)
plt.show()

# Calculate cumulated histogram
ch = ip.cumulated_histogram(matrix_gray, 8)
print("Gray image histogram & cumulated histogram:")
plt.plot(ch)
plt.show()

# Calculate an image from normalized cumulated histogram
equalized_matrix = ip.equalized_matrix(matrix_gray, 8)
print("Equalized image:", equalized_matrix.shape)
plt.imshow(equalized_matrix, cmap='gray')
plt.show()

# Calculate Equalized image histogram
eh = ip.histogram(equalized_matrix, 8)
plt.plot(eh)
ech = ip.cumulated_histogram(equalized_matrix, 8)
plt.plot(ech)
print("Equalized image histogram:")
plt.show()

# Save the image as png
# cv.imwrite("test_equalized_matrix.png", equalized_matrix)
