import cv2
import numpy as np

def SkinDetection(img, hist_data, thresh=0):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows,cols,channels = img_hsv.shape
    
    img_hsv_blank = np.copy(img_hsv)
    img_hsv_blank[:, :, :] = 0

    hist = hist_data['hist']
    h_edges = hist_data['h_edges']
    s_edges = hist_data['s_edges']

    for i in range(rows):
        for j in range(cols):
            h_ind = np.searchsorted(h_edges, img_hsv[i, j, 0])
            s_ind = np.searchsorted(s_edges, img_hsv[i, j, 1])
            h_ind = np.clip(h_ind, 0, hist.shape[0] - 1)
            s_ind = np.clip(s_ind, 0, hist.shape[1] - 1)
            if hist[h_ind, s_ind] > thresh:
                img_hsv_blank[i, j, :] = img_hsv[i, j, :]
    
    return cv2.cvtColor(img_hsv_blank, cv2.COLOR_HSV2BGR)