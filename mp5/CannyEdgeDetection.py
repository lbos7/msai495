import cv2
import numpy as np
from scipy.signal import convolve2d

def GaussSmoothing(img, N, Sigma):
    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    ax = np.linspace(-(N //2 ), N // 2, N)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * Sigma**2))
    kernel_norm = kernel / np.sum(kernel)
    img_smoothed = np.clip(convolve2d(img_gray, kernel_norm, mode='same', boundary='symm'), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_smoothed, cv2.COLOR_GRAY2BGR)

def ImageGradient(img_smoothed, N=5):
    img_gray = np.copy(img_smoothed)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(img_smoothed, cv2.CV_64F, 1, 0, ksize=N)
    gradient_y = cv2.Sobel(img_smoothed, cv2.CV_64F, 0, 1, ksize=N)
    mag = np.sqrt(gradient_x**2 + gradient_y**2)
    theta = np.arctan2(gradient_y, gradient_x)
    return mag,theta