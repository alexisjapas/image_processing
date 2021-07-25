#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:25:01 2021

@author: pouja
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# Convert a color matrix to gray
def color_to_gray(matrix, nbits):
    b, v, r = cv.split(matrix)
    matrix_gray = 0.299 * r + 0.587 * v + 0.114 * b

    return matrix_gray.astype(f'uint{nbits}')


def rgb_to_ycbcr(matrix, nbits):
    b, v, r = cv.split(matrix)
    y = 0.299 * r + 0.587 * v + 0.114 * b
    cb = 0.5 * b - 0.1687 * r - 0.3313 * v + 2**(nbits-1)
    cr = 0.5 * r - 0.4187 * v - 0.0813 * b + 2**(nbits-1)

    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr, nbits):
    matrix = np.zeros((y.shape[0], y.shape[1], 3))
    matrix[:, :, 0] = y + 1.772 * (cb - 2**(nbits-1))
    matrix[:, :, 1] = y - 0.34414 * (cb - 2**(nbits-1)) - 0.71414 * (cr - 2**(nbits-1))
    matrix[:, :, 2] = y + 1.402 * (cr - 2**(nbits-1))

    return matrix.astype(f'uint{nbits}')


# Calculate the histogram (normalized)
def histogram(matrix, nbits):
    hist = np.zeros(2**nbits)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            hist[int(matrix[i, j])] = hist[int(matrix[i, j])] + 1

    return hist / np.max(hist) * (2**nbits - 1)


# Calculate cumulated histogram (normalized)
def cumulated_histogram(matrix, nbits):
    hist = histogram(matrix, nbits)
    cumulated_hist = np.zeros(2**nbits)
    cumulated_hist[0] = hist[0]
    for i in range(1, hist.shape[0]):
        cumulated_hist[i] = cumulated_hist[i - 1] + hist[i]
    cumulated_hist /= np.max(cumulated_hist)

    return cumulated_hist * (2**nbits - 1)


def equalized_matrix(matrix, nbits):
    """Equalize entry matrix histogram and return equalized matrix"""
    new_matrix = matrix.copy()

    if len(matrix.shape) == 2:
        cumulated_hist = cumulated_histogram(matrix, nbits)
        new_matrix = np.array([[cumulated_hist[int(new_matrix[i, j])] for
                                j in range(new_matrix.shape[1])] for i in range(new_matrix.shape[0])])
    else:
        y, cb, cr = rgb_to_ycbcr(matrix, nbits)
        ey = equalized_matrix(y, nbits)
        new_matrix = ycbcr_to_rgb(ey, cb, cr, nbits)

    return new_matrix.astype(f'uint{nbits}')


def pixelized_matrix(matrix, nb_vertical_pixel, nb_horizontal_pixel):
    new_matrix = np.zeros_like(matrix)

    height = new_matrix.shape[0]
    width = new_matrix.shape[1]

    pixel_height = height / nb_vertical_pixel
    pixel_width = width / nb_horizontal_pixel

    # Scalar if shades of gray and array if multi-channel
    channels = 1 if len(new_matrix.shape) == 2 else new_matrix.shape[2]
    print(channels)

    # Pixelize entire squares
    if channels == 1:
        for i in range(nb_vertical_pixel):
            for j in range(nb_horizontal_pixel):
                start_i = int(i * pixel_height)
                end_i = int((i + 1) * pixel_height)
                start_j = int(j * pixel_width)
                end_j = int((j + 1) * pixel_width)
                new_matrix[start_i:end_i, start_j:end_j] = np.mean(matrix[start_i:end_i, start_j:end_j])
    else:
        for i in range(nb_vertical_pixel):
            for j in range(nb_horizontal_pixel):
                start_i = int(i * pixel_height)
                end_i = int((i + 1) * pixel_height)
                start_j = int(j * pixel_width)
                end_j = int((j + 1) * pixel_width)
                for c in range(channels):
                    new_matrix[start_i:end_i, start_j:end_j, c] = np.mean(matrix[start_i:end_i, start_j:end_j, c])

    return new_matrix.astype(np.uint8)
