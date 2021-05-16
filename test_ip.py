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
matrix = cv.imread("data/lenna.png")

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
egalized_matrix = ip.egalized_matrix(matrix_gray, ch)
plt.imshow(egalized_matrix, cmap='gray')
print("Egalized image:")
plt.show()

# Calculate egalized image histogram
eh = ip.histogram(egalized_matrix)
plt.plot(eh)
ech = ip.cumulated_histogram(eh)
plt.plot(ech)
print("Egalized image histogram:")
plt.show()

# Save the image as png
#cv.imwrite("test_egalized_matrix.png", egalized_matrix)