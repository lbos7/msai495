import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistoEqualization(img):

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    img_gray_1d = img_gray.flatten()
    hist = np.histogram(img_gray_1d, bins=256)
    
    plt.figure(1)
    plt.hist(img_gray_1d, bins=256)
    plt.title("moon.bmp histogram")
    plt.show()