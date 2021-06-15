#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:25:01 2021

@author: pouja
"""

import cv2 as cv
import numpy as np
import math as m


# Convert a color matrix to gray
def color_to_gray(matrix):
    b, v, r = cv.split(matrix)
    matrix_gray = 0.299*r + 0.587*v + 0.114*b
    return matrix_gray.astype(np.uint8)


# Calculate the histogram (normalized)
def histogram(matrix):
    hist = np.zeros(256, int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            hist[matrix[i, j]] = hist[matrix[i, j]] + 1
    return hist / np.max(hist) * 256


# Calculate cumulated histogram (normalized)
def cumulated_histogram(hist):
    cumulated_hist = np.zeros(256, int)
    cumulated_hist[0] = hist[0]
    for i in range(1, hist.shape[0]):
        cumulated_hist[i] = cumulated_hist[i - 1] + hist[i]
    return cumulated_hist / np.max(cumulated_hist) * 256


# Calculate new image matrix
def equalized_matrix(matrix, cumulated_hist):
    new_matrix = matrix.copy()
    for i in range(new_matrix.shape[0]):
        for j in range(new_matrix.shape[1]):
            new_matrix[i, j] = cumulated_hist[new_matrix[i, j]]
    return new_matrix


def pixelized_matrix(matrix, kernel_size):
    new_matrix = matrix.copy()
    for i in range(0, matrix.shape[0]-kernel_size+1, kernel_size):
        for j in range(0, matrix.shape[1]-kernel_size+1, kernel_size):
            medium = np.zeros(3)
            for k in range(kernel_size):
                for n in range(kernel_size):
                    medium += matrix[i+k][j+n]
            medium /= int(m.pow(kernel_size, 2))
            for k in range(kernel_size):
                for n in range(kernel_size):
                    new_matrix[i+k][j+n] = medium

    return new_matrix
