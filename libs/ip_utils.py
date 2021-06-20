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
    matrix_gray = 0.299 * r + 0.587 * v + 0.114 * b
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
    new_matrix = np.array([[cumulated_hist[new_matrix[i, j]] for
                            j in range(new_matrix.shape[1])] for i in range(new_matrix.shape[0])])
    return new_matrix


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
                    new_matrix[start_i:end_i, start_j:end_j, c] = np.mean(
                        matrix[start_i:end_i, start_j:end_j, c])

    return new_matrix
