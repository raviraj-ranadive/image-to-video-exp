import numpy as np
from PIL import Image
import cv2


coords  =  np.load("coords_64.npy")
print(coords.shape)
video_frames  = []
video = cv2.VideoCapture("video.mp4")


for i in range(20):
    ret, frame = video.read()
    video_frames.append(frame)
    if not ret:
        break
print("frames size " , len(video_frames), video_frames[0].shape)
#write point on each frame using coords and save video
all_frames = []
for i in range(20):
    for coord in coords:
        point  = coord[i]
        # print(point)
        cv2.circle(video_frames[i], (point[0], point[1]), 5, (0, 0, 255), -1)

    cv2.imwrite(f"./frames/frame_{i}.jpg", video_frames[i])
    all_frames.append(cv2.imread(f"./frames/frame_{i}.jpg"))


#make video from frames in mp4 
height, width, layers = all_frames[0].shape
size = (width,height)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)





