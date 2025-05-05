import cv2
import numpy as np
from CannyEdgeDetection import CannyEdgeDetection
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, center_of_mass

def HoughTransform(img, param_quant=180, N=5, sigma=1, percentageOfNonEdge=.85, max_filt_N=15, peaks_thresh_mul=.5):
    '''
    Detects prominent lines in a given image.
    
    Parameters:
        img (numpy array): The input image.
        param_quant (int): The number of parameter test points.
        N (int): The kernel size for smoothing.
        sigma (int): The standard deviation for smoothing.
        percentageOfNonEdge (float): Percentage of pixels that are not part of an edge.
        max_filt_N (int): Neighborhodd size for maximum filter.
        peaks_thresh_mul (float): Threshold multiplier for max filter peaks mask.
        
    Returns:
        (numpy array): The input image with the detected prominent lines.
    '''

    # Copying image, finding shape of imaage, and performing edge detection
    img_copy = np.copy(img)
    img_edges = CannyEdgeDetection(img, N=N, sigma=sigma, percentageOfNonEdge=percentageOfNonEdge)
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_BGR2GRAY)
    rows,cols = img_edges.shape

    # Finding offsets for range of possible parameter values based on image size
    rho_offset = int((rows ** 2 + cols ** 2) ** .5)
    theta_offset = 90

    # Setting up accumulator matrix
    accum_mat = np.zeros((2 * theta_offset + 1, 2 * rho_offset + 1))
    accum_mat = accum_mat.astype(np.int16)

    # Generating list of edge pixel coordinates
    edge_coords = np.argwhere(img_edges == 255)

    # Setting up list ot angles to check based on param_quant variable
    theta_check = np.linspace(-theta_offset, theta_offset, num=param_quant, dtype=np.int16)

    # Mapping from row,col to theta,rho
    for coord in edge_coords:
        row = coord[0]
        col = coord[1]

        # Calcutating rho using the current row,col coordinates with each angle in theta_check
        for theta in theta_check:
            rho = int(row * np.cos(np.deg2rad(theta)) + col * np.sin(np.deg2rad(theta)))
            accum_mat[theta + theta_offset, rho + rho_offset] += 1

    max_accum_val = np.max(accum_mat)

    # Apply maximum filter to find local maxima
    filtered = maximum_filter(accum_mat, size=max_filt_N)  # adjust size as needed
    peaks_mask = (accum_mat == filtered) & (accum_mat > max_accum_val*peaks_thresh_mul)

    # Label and get center of each peak
    labeled,_ = label(peaks_mask)
    peaks = center_of_mass(accum_mat, labeled, range(1, labeled.max()+1))

    # Converting peaks to lines in image
    for peak in peaks:

        # Extracting parameters
        theta = peak[0] - theta_offset
        rho = peak[1] - rho_offset

        # Converting parameters to row,col
        i0 = rho * np.cos(np.deg2rad(theta))
        j0 = rho * np.sin(np.deg2rad(theta))

        # Finding endpoints for line
        i1 = int(i0 - 1000 * np.sin(np.deg2rad(theta)))
        j1 = int(j0 + 1000 * np.cos(np.deg2rad(theta)))
        i2 = int(i0 + 1000 * np.sin(np.deg2rad(theta)))
        j2 = int(j0 - 1000 * np.cos(np.deg2rad(theta)))

        # Adding red line to image
        cv2.line(img_copy, (j1, i1), (j2, i2), (0, 0, 255), 2)

    return img_copy