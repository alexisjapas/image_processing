#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:25:01 2021

@author: pouja
"""

import cv2 as cv
import numpy as np


# Convert a color matrix to gray
def color_to_gray(matrix):
    b, v, r = cv.split(matrix)
    matrix_gray = 0.299*r + 0.587*v + 0.114*b
    return matrix_gray.astype(np.uint8)

# Caculate the histogram (normalized)
def histogram(matrix):
    histogram = np.zeros(256, int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            histogram[matrix[i, j]] = histogram[matrix[i, j]] + 1
    return histogram / np.max(histogram) * 256

# Caculate cumulated histogram (normalized)
def cumulated_histogram(histogram):
    cumulated_histogram = np.zeros(256, int)
    cumulated_histogram[0] = histogram[0]
    for i in range(1, histogram.shape[0]):
        cumulated_histogram[i] = cumulated_histogram[i-1] + histogram[i]
    return cumulated_histogram / np.max(cumulated_histogram) * 256

# Calculate 
def 

#min_index = 0
#min_val = hist_cumule[0]
#max_index = 0
#max_val = hist_cumule[0]
#for i in range(hist_cumule.size):
#    if hist_cumule[i] == min_val:
#        min_index = i
#    print(hist_cumule[i], max_val)
#    if hist_cumule[i] != max_val:
#        max_index = i
#        max_val = hist_cumule[i]
#
#print(min_index)
#print(max_index)
#
#
## Image finale
#plt.imshow(ng_mat, cmap='gray')
#plt.show()
#for i in range(ng_mat.shape[0]):
#    for j in range(ng_mat.shape[1]):
#        ng_mat[i, j] = hist_cumule[ng_mat[i, j]]
#
#plt.imshow(ng_mat, cmap='gray')
#plt.show()