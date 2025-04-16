import cv2
import numpy as np

def Dilation(img, se):
    
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    se_offset_rows = se[0] // 2
    se_offset_cols = se[1] // 2

    white_pix_coords = np.argwhere(img_gray == 255)

    for coord in white_pix_coords:
        i = coord[0]
        j = coord[1]
        for row in np.clip(np.linspace(i - se_offset_rows, i + se_offset_rows, se[0], dtype=int), 0, rows - 1):
                for col in np.clip(np.linspace(j - se_offset_cols, j + se_offset_cols, se[1], dtype=int), 0, cols - 1):
                    img_gray[row, col] = 255
    return img_gray


    return

def Erosion(img, se):
    return

def Opening(img, se):
    return

def Closing(img, se):
    return

def Boundary(img, se):
    return