import cv2
from HistoEqualization import HistoEqualization

moon_img = cv2.imread('mp3/images/moon.bmp')

moon_he_img = HistoEqualization(moon_img)

cv2.imshow('moon.bmp', moon_img)
cv2.imshow('HistoEqualization on moon.bmp', moon_he_img)
cv2.waitKey(0)
cv2.destroyAllWindows()