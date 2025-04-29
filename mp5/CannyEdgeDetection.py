import cv2
import numpy as np
from scipy.signal import convolve2d

def GaussSmoothing(img, N, Sigma):
    ax = np.linspace(-(N //2 ), N // 2, N)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * Sigma**2))
    kernel_norm = kernel / np.sum(kernel)
    
    return convolve2d(img, kernel_norm, mode='same', boundary='symm')