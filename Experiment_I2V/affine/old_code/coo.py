import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import cv2

# Generate random coordinates
coords = np.load("./coords.npy")

# Create 20 frames
frames = []
w,h = 1024,512
for frame_index in range(20):
    #use cv2 to draw points on each frame
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for point in coords[frame_index]:
        cv2.circle(frame, (point[0], point[1]), 5, (255, 0, 0), -1)
    frames.append(frame)


# Create video


clip = mpy.ImageSequenceClip([img[:, :,] for img in frames], fps=6)  # Convert RGB to BGR for moviepy
clip.write_videofile("project.mp4", codec="libx264")



