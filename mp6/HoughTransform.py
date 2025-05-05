import cv2
import numpy as np
from CannyEdgeDetection import CannyEdgeDetection
import matplotlib.pyplot as plt


def HoughTransform(img, param_quant=180, N=5, sigma=1, percentageOfNonEdge=.85):

    img_copy = np.copy(img)
    img_edges = CannyEdgeDetection(img, N=N, sigma=sigma, percentageOfNonEdge=percentageOfNonEdge)
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)

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

    max_accum_val = np.max(accum_mat)
    line_coords = np.argwhere(accum_mat > max_accum_val * .5)

    for line in line_coords:
        theta = line[0] - theta_offset
        rho = line[1] - rho_offset

        i0 = rho * np.cos(np.deg2rad(theta))
        j0 = rho * np.sin(np.deg2rad(theta))

        i1 = int(i0 - 1000 * np.sin(np.deg2rad(theta)))
        j1 = int(j0 + 1000 * np.cos(np.deg2rad(theta)))

        i2 = int(i0 + 1000 * np.sin(np.deg2rad(theta)))
        j2 = int(j0 - 1000 * np.cos(np.deg2rad(theta)))

        cv2.line(img_copy, (i1, j1), (i2, j2), (0, 0, 255), 2)

    return img_copy
        
    # Normalize the accumulator matrix and convert to uint8
    # accum_mat_norm = cv2.normalize(accum_mat, None, 0, 255, cv2.NORM_MINMAX)
    # accum_mat_uint8 = accum_mat_norm.astype(np.uint8)

    # # Apply histogram equalization
    # equ = cv2.equalizeHist(accum_mat_uint8)

    # print(np.unique(accum_mat)[::-1])  # Descending order

    # plt.figure()
    # plt.imshow(img_edges, cmap='gray')
    # plt.figure()
    # plt.imshow(accum_mat_norm, cmap='gray', aspect='auto')
    # plt.show()