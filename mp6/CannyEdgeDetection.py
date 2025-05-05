import cv2
import numpy as np
from scipy.signal import convolve2d

def GaussSmoothing(img, N=5, sigma=1):
    '''
    Applies the Gaussian smoothing to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        N (int): The kernel size for smoothing.
        sigma (int): The standard deviation for smoothing.
        
    Returns:
        (numpy array): The smoothed image.
    '''

    # Converting image to grayscale
    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    # Generating mesh grid
    ax = np.linspace(-(N // 2), N // 2, N)
    xx, yy = np.meshgrid(ax, ax)

    # Finding normalized kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel_norm = kernel / np.sum(kernel)

    # Smoothing image
    img_smoothed = np.clip(convolve2d(img_gray, kernel_norm, mode='same', boundary='symm'), 0, 255).astype(np.uint8)

    return cv2.cvtColor(img_smoothed, cv2.COLOR_GRAY2BGR)


def ImageGradient(img_smoothed, N=5):
    '''
    Finds the image gradient for a given image.
    
    Parameters:
        img_smoothed (numpy array): An image that has been smoothed using Gaussian smoothing.
        N (int): The kernel size.
        
    Returns:
        mag (numpy array): An array representing the image gradient magnitude at each pixel.
        theta (numpy array): An array representing the image gradient angle at each pixel.
    '''

    # Converting image to grayscale
    img_gray = np.copy(img_smoothed)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    # Calculating gradient in x and y directions
    gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=N)
    gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=N)

    # Calculating outputs
    mag = np.sqrt(gradient_x**2 + gradient_y**2)
    theta = np.arctan2(gradient_y, gradient_x)

    return mag,theta


def FindThreshold(mag, percentageOfNonEdge=.8):
    '''
    Finds thresholds based on magnitude array of an image gradient.
    
    Parameters:
        mag (numpy array): An array representing the image gradient magnitude at each pixel.
        percentageOfNonEdge (float): Percentage of pixels that are not part of an edge.
        
    Returns:
        T_high (float): High threshold value for pixel magnitudes.
        T_low (float): Low threshold value for pixel magnitudes.
    '''

    # Generating histogram based on magnitude array
    hist,bins = np.histogram(mag.flatten(), bins=int(np.max(mag)), density=True)

    # Finding cdf based on histogram
    cdf = np.cumsum(hist)

    # Finding threshold values
    ind = np.where(cdf > percentageOfNonEdge)[0][0]
    T_high = bins[ind]
    T_low = 0.5 * T_high

    return T_high,T_low


def NonmaximaSuppress(mag, theta):
    '''
    Suppresses pixel magnitudes that are not maximas.
    
    Parameters:
        mag (numpy array): An array representing the image gradient magnitude at each pixel.
        theta (numpy array): An array representing the image gradient angle at each pixel.
        
    Returns:
        mag_suppressed (numpy array): An array representing the image gradient magnitude at each pixel with non-maximas suppressed.
    '''

    # Checking shape of mag array and setting up empty array
    rows,cols = mag.shape
    mag_suppressed = np.zeros((rows, cols))

    # Adding padding to make it easier to check border of array
    mag_padded = np.pad(mag, pad_width=1, mode='edge')
    theta_padded = np.pad(theta, pad_width=1, mode='edge')

    # Quantize angles to 4 main directions
    angle = theta_padded * 180. / np.pi
    angle[angle < 0] += 180

    # Suppressing non-maximas
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            q = 0
            r = 0

            # angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag_padded[i, j+1]
                r = mag_padded[i, j-1]
            # angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag_padded[i+1, j-1]
                r = mag_padded[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag_padded[i+1, j]
                r = mag_padded[i-1, j]
            # angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag_padded[i-1, j-1]
                r = mag_padded[i+1, j+1]

            # Keep pixel if it's a local maximum
            if (mag_padded[i,j] >= q) and (mag_padded[i,j] >= r):
                mag_suppressed[i - 1, j - 1] = mag_padded[i,j]
            else:
                mag_suppressed[i - 1, j - 1] = 0

    return mag_suppressed


def EdgeLinking(mag_low, mag_high):
    '''
    Links edges that have been detected on an image.
    
    Parameters:
        mag_low (numpy array): An array representing an image with the edges greater than a low threshold.
        mag_high (numpy array): An array representing an image with the edges greater than a high threshold.
        
    Returns:
        edges (numpy array): An array representing an image with edges detected.
    '''

    rows, cols = mag_high.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)

    # Iterate over each strong edge pixel
    for i in range(rows):
        for j in range(cols):
            if mag_high[i, j] and not visited[i, j]:
                # Use a stack instead of recursion
                stack = [(i, j)]

                while stack:
                    ci, cj = stack.pop()

                    if visited[ci, cj]:
                        continue

                    visited[ci, cj] = True
                    edges[ci, cj] = 255

                    # Check 8-connected neighbors
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = ci + di, cj + dj
                            if (0 <= ni < rows and 0 <= nj < cols and 
                                not visited[ni, nj] and mag_low[ni, nj]):
                                stack.append((ni, nj))

    return edges


def CannyEdgeDetection(img, N=5, sigma=1, percentageOfNonEdge=.8):
    '''
    Performs Canny edge detection on an image.
    
    Parameters:
        img (numpy array): The input image.
        N (int): The kernel size for smoothing.
        sigma (int): The standard deviation for smoothing.
        percentageOfNonEdge (float): Percentage of pixels that are not part of an edge.
        
    Returns:
        (numpy array): The input image with edges detected.
    '''

    # Smoothing
    img_smoothed = GaussSmoothing(img, N=N, sigma=sigma)

    # Finding gradient
    mag,theta = ImageGradient(img_smoothed, N=N)

    # Determining threshold values
    T_high,T_low = FindThreshold(mag, percentageOfNonEdge=percentageOfNonEdge)

    # Suppressing non-maximas
    mag_suppressed = NonmaximaSuppress(mag, theta)

    # Fiding edges that are greater than both threshold values
    mag_high = (mag_suppressed >= T_high).astype(np.uint8)
    mag_low = (mag_suppressed >= T_low).astype(np.uint8)

    # Linking edges
    edges = EdgeLinking(mag_low, mag_high)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)