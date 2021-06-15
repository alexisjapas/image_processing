#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:04:07 2021

@author: pouja
"""

import matplotlib.pyplot as plt
import cv2 as cv

import libs.ip_utils as ip


# Colored matrix
matrix = cv.imread("data/kolibri.jpg")

# Let's move to gray scale
matrix_gray = ip.color_to_gray(matrix)
plt.imshow(matrix_gray, cmap='gray')
print("Gray image:")
plt.show()

# Calculate it's histogram
h = ip.histogram(matrix_gray)
plt.plot(h)

# Calculate cumulated histogram
ch = ip.cumulated_histogram(h)
plt.plot(ch)
print("Gray image histogram & cumulated histogram:")
plt.show()

# Calculate an image from normalized cumulated histogram
equalized_matrix = ip.equalized_matrix(matrix_gray, ch)
plt.imshow(equalized_matrix, cmap='gray')
print("Equalized image:")
plt.show()

# Calculate Equalized image histogram
eh = ip.histogram(equalized_matrix)
plt.plot(eh)
ech = ip.cumulated_histogram(eh)
plt.plot(ech)
print("Equalized image histogram:")
plt.show()

# Save the image as png
# cv.imwrite("test_equalized_matrix.png", equalized_matrix)
