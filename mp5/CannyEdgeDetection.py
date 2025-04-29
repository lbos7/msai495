import cv2
import numpy as np
from scipy.signal import convolve2d

def GaussSmoothing(img, N, Sigma):
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    ax = np.linspace(-(N //2 ), N // 2, N)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * Sigma**2))
    kernel_norm = kernel / np.sum(kernel)
    img_smoothed = np.clip(convolve2d(img_gray, kernel_norm, mode='same', boundary='symm'), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_smoothed, cv2.COLOR_GRAY2BGR)