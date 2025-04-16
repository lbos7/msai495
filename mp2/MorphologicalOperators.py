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
        for row in np.unique(np.clip(np.linspace(i - se_offset_rows, i + se_offset_rows, se[0], dtype=int), 0, rows - 1)):
                for col in np.unique(np.clip(np.linspace(j - se_offset_cols, j + se_offset_cols, se[1], dtype=int), 0, cols - 1)):
                    img_gray[row, col] = 255
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

def Erosion(img, se):
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    se_offset_rows = se[0] // 2
    se_offset_cols = se[1] // 2

    keep_coords = []

    white_pix_coords = np.argwhere(img_gray == 255)

    for coord in white_pix_coords:
        i = coord[0]
        j = coord[1]
        sum = 0
        for row in np.unique(np.clip(np.linspace(i - se_offset_rows, i + se_offset_rows, se[0], dtype=int), 0, rows - 1)):
            for col in np.unique(np.clip(np.linspace(j - se_offset_cols, j + se_offset_cols, se[1], dtype=int), 0, cols - 1)):
                sum += img_gray[row, col]
        if sum == 255*se[0]*se[1]:
            keep_coords.append(coord)

    for coord in white_pix_coords:
        keep = False
        for keep_coord in keep_coords:
            if np.array_equal(coord, keep_coord):
                keep = True
                break
        if not keep:
            img_gray[coord[0], coord[1]] = 0
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

def Opening(img, se):
    eroded_img = Erosion(img, se)
    return Dilation(eroded_img, se)

def Closing(img, se):
    dilated_img = Dilation(img, se)
    return Erosion(dilated_img, se)

def Boundary(img, se):
    eroded_img = Erosion(img, se)
    return img - eroded_img