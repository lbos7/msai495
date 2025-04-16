import cv2
import numpy as np
from MorphologicalOperators import Dilation, Erosion, Opening, Closing, Boundary

gun_img = cv2.imread('mp2/images/gun.bmp')

se = (3, 3)
gun_img_dilated = Dilation(gun_img, se)
gun_img_eroded = Erosion(gun_img_dilated, se)
gun_img_opened = Opening(gun_img, se)
gun_img_closed = Closing(gun_img, se)
gun_img_boundary = Boundary(gun_img_eroded, se)

cv2.imshow('gun.bmp', gun_img)
cv2.imshow('dilated gun.bmp', gun_img_dilated)
cv2.imshow('eroded gun.bmp', gun_img_eroded)
cv2.imshow('opened gun.bmp', gun_img_opened)
cv2.imshow('closed gun.bmp', gun_img_closed)
cv2.imshow('boundary of gun.bmp', gun_img_boundary)
cv2.waitKey(0)
cv2.destroyAllWindows()