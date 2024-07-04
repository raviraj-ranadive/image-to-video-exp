import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_zoom_parameters(coords_prev, coords_current):
    """
    Calculate the average movement vector and the center of movement.
    coords_prev: [N, 2] array of coordinates in the previous frame
    coords_current: [N, 2] array of coordinates in the current frame
    Returns the center of zoom and scale factor.
    """
    # Ensure the coordinates are in floating point for accurate calculations
    coords_prev = coords_prev.astype(np.float64)
    coords_current = coords_current.astype(np.float64)

    # Calculate movement vectors
    movement_vectors = coords_current - coords_prev
    average_vector = np.mean(movement_vectors, axis=0)
    magnitude = np.linalg.norm(average_vector)
    center_of_movement = np.mean(coords_current, axis=0)

    # Determine zoom direction and scale
    if magnitude > 0.1:  # Lower threshold for subtle movements
        scale = 1 + magnitude / 100  # Scale adjustment based on subtle movement magnitude
    else:
        scale = 1

    return center_of_movement, scale

def zoom_image(img, center, scale):
    """
    Applies zoom to the image based on center and scale.
    """
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, 0, scale)
    zoomed_img = cv2.warpAffine(img, M, (width, height))
    return zoomed_img

# Load an image
image = cv2.imread('./product_white.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load coordinates
coords = np.load("./coords.npy")
print("coords shape ",coords.shape)
# coords shape  (20, 1024, 2) so take points for each frame at  355 and 277 from each frame
coords = coords[:, [355, 277], :]

# Apply dynamic zoom based on coordinates
images = []
images.append(image)
for i in range(1, len(coords)):
    center, scale = calculate_zoom_parameters(coords[i-1], coords[i])
    # print("CoTrack Coords",coords[i-1], coords[i])
    # print("calculate_zoom_parameters  center -> ",center,  "scale -> ",scale)
    zoomed = zoom_image(image, (int(center[0]), int(center[1])), scale)
    images.append(zoomed)

print("images len ",len(images))
# Visualize the result for some frames
# fig, axes = plt.subplots(1, 5, figsize=(20, 6))
# for i, ax in enumerate(axes):
#     ax.imshow(images[i])
#     ax.axis('off')
#     ax.set_title(f'Frame {i}')
# plt.show()
import moviepy.editor as mpy
import os

#create video mp4 
clip = mpy.ImageSequenceClip(images, fps=15)

clip.write_videofile("project.mp4")
