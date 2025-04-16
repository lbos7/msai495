import cv2
import numpy as np
from MorphologicalOperators import Dilation, Erosion

gun_img = cv2.imread('mp2/images/gun.bmp')

se = (3, 3)
gun_img_dilated = Dilation(gun_img, se)
gun_img_eroded = Erosion(gun_img, se)

cv2.imshow('gun.bmp', gun_img)
cv2.imshow('dilated gun.bmp', gun_img_dilated)
cv2.imshow('eroded gun.bmp', gun_img_eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()