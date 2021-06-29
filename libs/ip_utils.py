#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:25:01 2021

@author: pouja
"""

import cv2 as cv
import numpy as np


# Convert a color matrix to gray
def color_to_gray(matrix, nbits):
    b, v, r = cv.split(matrix)
    matrix_gray = 0.299 * r + 0.587 * v + 0.114 * b

    return matrix_gray.astype(f'uint{nbits}')


# Calculate the histogram (normalized)
def histogram(matrix, nbits):
    channels = 1 if len(matrix.shape) == 2 else matrix.shape[2]
    if channels == 1:
        hist = np.zeros(2**nbits)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                hist[matrix[i, j]] = hist[matrix[i, j]] + 1
    else:
        hist = np.zeros((2**nbits, channels))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                for c in range(channels):
                    hist[matrix[i, j, c], c] = hist[matrix[i, j, c], c] + 1

    return hist / np.max(hist) * (2**nbits - 1)


# Calculate cumulated histogram (normalized)
def cumulated_histogram(matrix, nbits):
    hist = histogram(matrix, nbits)
    channels = 1 if len(hist.shape) == 1 else hist.shape[1]
    if channels == 1:
        cumulated_hist = np.zeros(2**nbits)
        cumulated_hist[0] = hist[0]
        for i in range(1, hist.shape[0]):
            cumulated_hist[i] = cumulated_hist[i - 1] + hist[i]
        cumulated_hist /= np.max(cumulated_hist)
    else:
        cumulated_hist = np.zeros((2**nbits, hist.shape[1]))
        cumulated_hist[0] = hist[0]
        for i in range(1, hist.shape[0]):
            cumulated_hist[i] = cumulated_hist[i - 1] + hist[i]
        cumulated_hist /= np.amax(cumulated_hist)

    return cumulated_hist * (2**nbits - 1)


def equalized_matrix(matrix, nbits):
    """Equalize entry matrix histogram and return equalized matrix"""
    cumulated_hist = cumulated_histogram(matrix, nbits)

    new_matrix = matrix.copy()
    channels = 1 if len(matrix.shape) == 2 else matrix.shape[2]
    if channels == 1:
        new_matrix = np.array([[cumulated_hist[new_matrix[i, j]] for
                                j in range(new_matrix.shape[1])] for i in range(new_matrix.shape[0])])
    else:
        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                for c in range(channels):
                    new_matrix[i, j, c] = cumulated_hist[new_matrix[i, j, c], c]

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
