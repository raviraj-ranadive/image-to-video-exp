import numpy as np
import cv2
import os
import moviepy.editor as mpy

width, height = 1024, 576

# Function to apply scaling
def scale_image(image, scale_factor):
    M = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0]])
    scaled_image = cv2.warpAffine(image, M, (width, height))
    return scaled_image
    

# Function to apply transformation based on srcTri and dstTri
def apply_transformation(image, srcTri, dstTri, scale_factor=1.0):
    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # Apply the affine transformation to the image
    transformed_image = cv2.warpAffine(image, M, (width, height))
    # Apply scaling
    transformed_image = scale_image(transformed_image, scale_factor)
    return transformed_image


# Load the image and points
image = cv2.imread('/Users/ameerazam/Documents/affine/product.png', cv2.IMREAD_UNCHANGED)
points = np.load('/Users/ameerazam/Documents/affine/coords.npy')

# Define points
pts1 = 298  # 10
pts2 = 684  # 200
pts3 = 205  # 720


srcTri = np.array([[points[0][pts1][0], points[0][pts1][1]],
                   [points[0][pts2][0], points[0][pts2][1]],
                   [points[0][pts3][0], points[0][pts3][1]]], dtype=np.float32)

# Calculate scale factor



final_frame = []

for index in range(1, len(points) - 1):
    dstTri = np.array([[points[index][pts1][0], points[index][pts1][1]],
                       [points[index][pts2][0], points[index][pts2][1]],
                       [points[index][pts3][0], points[index][pts3][1]]], dtype=np.float32)

    # Apply transformations with calculated scale factor
    initial_distance = np.linalg.norm(srcTri[0] - srcTri[1])

    scale_factor = width / initial_distance
    transformed_image = apply_transformation(image, srcTri, dstTri, scale_factor=scale_factor)

    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        transformed_image = cv2.circle(transformed_image, (int(dstTri[i][0]), int(dstTri[i][1])), 5, (255, 0, 0), -1)
    
    final_frame.append(transformed_image)

# Create and save the video
clip = mpy.ImageSequenceClip(final_frame, fps=6)
clip.write_videofile('check.mp4')
