import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

cropped_images_path = 'mp4/images/cropped_images/'
cropped_images_folder = Path(cropped_images_path)


h_vals = []
s_vals = []

for f in cropped_images_folder.iterdir():
    if f.is_file():

        img = cv2.imread(cropped_images_path + f.name)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_vals.extend(img_hsv[:, :, 0].flatten())
        s_vals.extend(img_hsv[:, :, 1].flatten())

hist, xedges, yedges = np.histogram2d(h_vals, s_vals, bins=[180, 256], range=[[0, 179], [0, 255]], density=True)

plt.figure(1)
plt.imshow(hist.T, interpolation='nearest', origin='lower', aspect='auto')
plt.colorbar(label='Probability')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.title('2D Histogram of Hue and Saturation')
plt.show()