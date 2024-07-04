import cv2
import numpy as np


arr1 = np.load("coords_64.npy")
print(arr1.shape)
arr2  = np.load("pred_tracks_64.npy")

print("min ", np.min(arr1), "max " ,  np.min(arr1))
print("max ", np.max(arr2), np.max(arr2))
