import numpy as np
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy

# def calculate_zoom_parameters(coords_prev, coords_current):
#     """
#     Calculate the average movement vector and the center of movement.
#     Adjust zoom scale based on the movement direction and magnitude.
#     """
#     coords_prev = coords_prev.astype(np.float64)
#     coords_current = coords_current.astype(np.float64)

#     # Calculate movement vectors
#     movement_vectors = coords_current - coords_prev
#     average_vector = np.mean(movement_vectors, axis=0)
#     magnitude = np.linalg.norm(average_vector)
#     center_of_movement = np.mean(coords_current, axis=0)

#     # Direction of movement - zoom in if converging, zoom out if diverging
#     direction_converging = np.all(movement_vectors < 0) or np.all(movement_vectors > 0)

#     sensitive  =   50
#     if magnitude > 0.1:
#         if direction_converging:
#             scale = 1 + magnitude / sensitive
#         else:
#             scale = max(0.5, 1 - magnitude / sensitive)  # prevent scale from being negative
#     else:
#         scale = 1

#     return center_of_movement, scale

# def zoom_image(img, center, scale):
#     """
#     Applies zoom to the image based on center and scale.
#     """
#     height, width = img.shape[:2]
#     M = cv2.getRotationMatrix2D(center, 0, scale)
#     zoomed_img = cv2.warpAffine(img, M, (width, height))
#     return zoomed_img

def calculate_zoom_parameters(coords_prev, coords_current):
    coords_prev = coords_prev.astype(np.float64)
    coords_current = coords_current.astype(np.float64)
    movement_vectors = coords_current - coords_prev
    average_vector = np.mean(movement_vectors, axis=0)
    magnitude = np.linalg.norm(average_vector)
    center_of_movement = np.mean(coords_current, axis=0)

    # Exaggerate the scale changes
    if magnitude > 0.1:
        scale = 1 + magnitude / 50  # More sensitive zoom based on movement
    else:
        scale = 1

    return center_of_movement, scale

def zoom_image(img, center, scale):
    height, width = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, 0, scale)

    zoomed_img = cv2.warpAffine(img, M, (width, height))
    return zoomed_img

# Load an image
image = cv2.imread('./crocs1/product_crcos_main.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load coordinates
coords = np.load("./crocs1/coords.npy")
print("coords shape ",coords.shape)
# coords = coords[:, [355, 277], :]  # Select specific points as per the requirement

# Apply dynamic zoom based on coordinates
images = [image]  # Start with the initial image
for i in range(1, len(coords)):
    # print("coords[i-1] ",coords[i-1])
    # center, scale = calculate_zoom_parameters(coords[i-1], coords[i])
    center, scale = calculate_zoom_parameters(coords[0], coords[i])  # Calculate movement from the starting frame
    print("center -> ",center,  "scale -> ",scale)
    zoomed = zoom_image(image, (int(center[0]), int(center[1])), scale)
    images.append(zoomed)

    


# Create video from images using moviepy
clip = mpy.ImageSequenceClip([img[:, :,] for img in images], fps=6)  # Convert RGB to BGR for moviepy
clip.write_videofile("project_crocs1.mp4", codec="libx264")
