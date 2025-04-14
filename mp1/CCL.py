import cv2
import numpy as np

def CCL(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = img_gray.shape
