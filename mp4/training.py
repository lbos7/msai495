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

hist, h_edges, s_edges = np.histogram2d(h_vals, s_vals, bins=[180, 256], range=[[0, 179], [0, 255]])
log_hist = np.log1p(hist)

# Normalize for display
log_hist_norm = cv2.normalize(log_hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
log_hist_norm = log_hist_norm.astype(np.uint8)

plt.figure(1)
plt.imshow(log_hist_norm.T, interpolation='nearest', origin='lower', aspect='auto')
plt.colorbar(label='Normalized Value')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.title('2D Histogram of Hue and Saturation')
plt.show()

np.savez('skin_hist.npz', hist=log_hist_norm, h_edges=h_edges, s_edges=s_edges)