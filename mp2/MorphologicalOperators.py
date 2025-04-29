import cv2
import numpy as np

def Dilation(img, se):
    '''
    Applies the dilation morphological operator to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        se (tuple): The structuring element used in the operation.
        
    Returns:
        (numpy array): The dilated image.
    '''

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    # Determining the offsets from the center pixel of the structuring element
    se_offset_rows = se[0] // 2
    se_offset_cols = se[1] // 2

    # Adding padding for the image to account structuring element on the border of the image
    padded_img = np.pad(img_gray, ((se_offset_rows, se_offset_rows), (se_offset_cols, se_offset_cols)), mode="constant", constant_values=0)

    # Locating white pixels
    white_pix_coords = np.argwhere(padded_img == 255)

    # Dilation process
    for coord in white_pix_coords:
        i = coord[0]
        j = coord[1]
        row_inds = np.repeat(np.linspace(i - se_offset_rows, i + se_offset_rows, se[0], dtype=int), se[1])
        col_inds = np.tile(np.linspace(j - se_offset_cols, j + se_offset_cols, se[1], dtype=int), se[0])
        padded_img[row_inds, col_inds] = 255

    # Removing padding
    final_img = padded_img[se_offset_rows:(rows + se_offset_rows), se_offset_cols:(cols + se_offset_cols)]

    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def Erosion(img, se):
    '''
    Applies the erosion morphological operator to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        se (tuple): The structuring element used in the operation.
        
    Returns:
        (numpy array): The eroded image.
    '''

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    # Determining the offsets from the center pixel of the structuring element
    se_offset_rows = se[0] // 2
    se_offset_cols = se[1] // 2

    # Adding padding for the image to account structuring element on the border of the image
    padded_img = np.pad(img_gray, ((se_offset_rows, se_offset_rows), (se_offset_cols, se_offset_cols)), mode="constant", constant_values=0)

    # Copying padded image for overwriting later
    img_new = np.copy(padded_img)
    img_new[:, :] = 0

    # Locating white pixels
    white_pix_coords = np.argwhere(padded_img == 255)

    # Erosion process
    for coord in white_pix_coords:
        i = coord[0]
        j = coord[1]
        row_inds = np.repeat(np.linspace(i - se_offset_rows, i + se_offset_rows, se[0], dtype=int), se[1])
        col_inds = np.tile(np.linspace(j - se_offset_cols, j + se_offset_cols, se[1], dtype=int), se[0])
        sum = np.sum(padded_img[row_inds, col_inds], dtype=int)
        if sum == 255*se[0]*se[1]:
            img_new[i, j] = 255

    # Removing padding
    final_img = img_new[se_offset_rows:(rows + se_offset_rows), se_offset_cols:(cols + se_offset_cols)]

    return cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


def Opening(img, se):
    '''
    Applies the opening morphological operator to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        se (tuple): The structuring element used in the operation.
        
    Returns:
        (numpy array): The opened image.
    '''

    eroded_img = Erosion(img, se)
    
    return Dilation(eroded_img, se)


def Closing(img, se):
    '''
    Applies the closing morphological operator to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        se (tuple): The structuring element used in the operation.
        
    Returns:
        (numpy array): The closed image.
    '''

    dilated_img = Dilation(img, se)

    return Erosion(dilated_img, se)


def Boundary(img, se):
    '''
    Applies the boundary morphological operator to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        se (tuple): The structuring element used in the operation.
        
    Returns:
        (numpy array): The boundary image.
    '''

    eroded_img = Erosion(img, se)

    return img - eroded_img