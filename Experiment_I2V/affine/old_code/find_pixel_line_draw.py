import numpy as np
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy



coords  = np.load("crocs1/coords.npy")
print(coords.shape)

frames = []

# Total 20 frames in the video

center_point = (1024//2, 576//2)


all_points = []
w,h  =  576 , 1024


for frame_coords in coords:
    
    
    # Draw circles for each coordinate in the current frame
    for coord in frame_coords:
        point  =  coord[0], coord[1]
        print(point)
        all_points.append(point)
        break 

print("All points", np.array(all_points).shape)

# draw line on the image
for i in range(1, len(all_points)):
    dummy_white_image = np.ones((w, h, 3), dtype=np.uint8) * 255
    #draw center point
    dummy_white_image = cv2.circle(dummy_white_image, center_point, 10, (0, 0, 255), -1)

    dummy_white_image = cv2.line(dummy_white_image, all_points[i-1], all_points[i], (255, 0, 0), 2)
    frames.append(dummy_white_image)






clip = mpy.ImageSequenceClip(frames, fps=6)
clip.write_videofile("frames_draw_line.mp4")
