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
print(matrix_gray.size)
plt.imshow(matrix_gray, cmap='gray')
plt.show()

# Calculate it's histogram
h = ip.histogram(matrix_gray)
plt.plot(h)

# Calculate cumulated histogram
ch = ip.cumulated_histogram(h)
plt.plot(ch)
plt.show()

#cv.imwrite("test_enregistrement.png", matG2)