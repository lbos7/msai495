import cv2
import numpy as np
from CannyEdgeDetection import GaussSmoothing, ImageGradient, FindThreshold, CannyEdgeDetection
import matplotlib.pyplot as plt


def HoughTransform(img, param_quant=180, N=5, sigma=1, percentageOfNonEdge=.93):

    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    # Smoothing
    # img_smoothed = GaussSmoothing(img)

    # # Finding gradient
    # mag, theta = ImageGradient(img_smoothed)

    # # Determining threshold values
    # T_high, T_low = FindThreshold(mag)

    img_edges = CannyEdgeDetection(img, N=5, sigma=1, percentageOfNonEdge=.95)
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)

    # img_edges = cv2.Canny(img_gray, T_low, T_high)

    rows,cols = img_edges.shape

    rho_offset = int((rows ** 2 + cols ** 2) ** .5)
    theta_offset = 90
    accum_mat = np.zeros((2 * theta_offset + 1, 2 * rho_offset + 1))
    accum_mat = accum_mat.astype(np.int16)

    edge_coords = np.argwhere(img_edges == 255)

    theta_check = np.linspace(-theta_offset, theta_offset, num=param_quant, dtype=np.int16)

    for coord in edge_coords:
        row = coord[0]
        col = coord[1]

        for theta in theta_check:
            rho = int(row * np.cos(np.deg2rad(theta)) + col * np.sin(np.deg2rad(theta)))
            accum_mat[theta + theta_offset, rho + rho_offset] += 1

    # Normalize the accumulator matrix and convert to uint8
    accum_mat_norm = cv2.normalize(accum_mat, None, 0, 255, cv2.NORM_MINMAX)
    accum_mat_uint8 = accum_mat_norm.astype(np.uint8)

    # Apply histogram equalization
    equ = cv2.equalizeHist(accum_mat_uint8)

    plt.figure()
    plt.imshow(img_edges, cmap='gray')
    plt.figure()
    plt.imshow(accum_mat_norm, cmap='gray', aspect='auto')
    plt.show()
