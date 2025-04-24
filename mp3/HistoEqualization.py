import cv2
import numpy as np

def HistoEqualization(img):
    '''
    Applies the histogram equalization to a given image.
    
    Parameters:
        img (numpy array): The image the operation will be performed on.
        
    Returns:
        (numpy array): The processed image.
    '''

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    # Generating histogram and cdf
    hist,bins = np.histogram(img_gray.flatten(), bins=256, density=True)
    cdf = np.cumsum(hist)

    # Defining array for transforming input and output gray levels
    L2_arr = cdf*255
    
    # Editing pixel values based on transformation
    for i in range(rows):
        for j in range(cols):
            img_gray[i, j] = int(L2_arr[img_gray[i, j]])

    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)