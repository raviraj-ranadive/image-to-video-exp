import numpy as np 
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt


coords  =  np.load("/Users/ameerazam/Documents/affine/crocs1/coords.npy")
coords = coords - 100
base_mask= Image.open("/Users/ameerazam/Documents/affine/crocs1/crocs_mask_1.png")
mask_check= Image.open("/Users/ameerazam/Documents/affine/crocs1/crocs_mask_1.png").convert("RGB")
base_mask = np.array(base_mask)

print("coords ",coords.shape)





frames = []

points_white_area = []
for coord in coords[:1]:
    pts = coord
    #check if pts lies in the  white region of the mask
    # base_mask = cv.threshold(base_mask, 127, 1, cv.THRESH_BINARY)[1] 
    #multi with frames 
    for pt in pts:
        try:
            if base_mask[pt[1],pt[0]] == True:
                print("Point lies in the black region of the mask")
                dummy_frame =np.array(mask_check)
                dummy_frame = cv.circle(dummy_frame, (pt), 5, (255, 0, 255), -1)
                frames.append(dummy_frame)
                points_white_area.append(pt)
        except:
            print("Point lies in the black region of the mask")

print("points_white_area ",np.array(points_white_area).shape)

import moviepy.editor as mpy
clip = mpy.ImageSequenceClip(frames, fps=6)
clip.write_videofile('check-point.mp4')


