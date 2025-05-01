import cv2
import numpy as np
from scipy.signal import convolve2d

def GaussSmoothing(img, N=5, sigma=1):
    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    ax = np.linspace(-(N // 2), N // 2, N)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel_norm = kernel / np.sum(kernel)
    img_smoothed = np.clip(convolve2d(img_gray, kernel_norm, mode='same', boundary='symm'), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_smoothed, cv2.COLOR_GRAY2BGR)

def ImageGradient(img_smoothed, N=5):
    img_gray = np.copy(img_smoothed)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=N)
    gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=N)
    mag = np.sqrt(gradient_x**2 + gradient_y**2)
    theta = np.arctan2(gradient_y, gradient_x)
    return mag, theta

def FindThreshold(mag, percentageOfNonEdge=.8):
    hist,bins = np.histogram(mag.flatten(), bins=int(np.max(mag)), density=True)
    cdf = np.cumsum(hist)
    ind = np.where(cdf > percentageOfNonEdge)[0][0]
    T_high = bins[ind]
    T_low = 0.5 * T_high
    return T_high,T_low

def NonmaximaSuppress(mag, theta):
    rows,cols = mag.shape
    mag_suppressed = np.zeros((rows, cols))

    mag_padded = np.pad(mag, pad_width=1, mode='edge')
    theta_padded = np.pad(theta, pad_width=1, mode='edge')

    # Quantize angles to 4 main directions
    angle = theta_padded * 180. / np.pi
    angle[angle < 0] += 180

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
    rows,cols = mag_high.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(i, j):
        if visited[i, j]:
            return
        visited[i, j] = True
        edges[i, j] = 255

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if (0 <= ni < rows) and (0 <= nj < cols):
                    if not visited[ni, nj] and mag_low[ni, nj]:
                        dfs(ni, nj)

    # Start DFS from all strong edge pixels
    for i in range(rows):
        for j in range(cols):
            if mag_high[i, j] and not visited[i, j]:
                dfs(i, j)

    return edges

def CannyEdgeDetection(img, N=5, sigma=1, percentageOfNonEdge=.8):
    img_smoothed = GaussSmoothing(img, N=N, sigma=sigma)
    mag,theta = ImageGradient(img_smoothed, N=N)
    T_high,T_low = FindThreshold(mag, percentageOfNonEdge=percentageOfNonEdge)
    mag_suppressed = NonmaximaSuppress(mag, theta)
    mag_high = (mag_suppressed >= T_high).astype(np.uint8)
    mag_low = (mag_suppressed >= T_low).astype(np.uint8)
    edges = EdgeLinking(mag_low, mag_high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)