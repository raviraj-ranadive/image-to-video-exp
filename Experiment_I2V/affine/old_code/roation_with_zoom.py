# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import moviepy.editor as mpy

# def calculate_affine_matrix(pts1, pts2):
#     """
#     Calculate an affine transformation matrix from pts1 to pts2.
#     """
#     if pts1.shape[0] == 3 and pts2.shape[0] == 3:  # Exactly three points
#         return cv2.getAffineTransform(pts1.astype(np.float32), pts2.astype(np.float32))
#     else:
#         # Compute the affine transformation with more points using least squares
#         return cv2.estimateAffinePartial2D(pts1.astype(np.float32), pts2.astype(np.float32))[0]

# def apply_transform(img, M):
#     """
#     Applies the affine transformation to the image.
#     """
#     height, width = img.shape[:2]
#     return cv2.warpAffine(img, M, (width, height))

# def compute_zoom_and_rotation(coords_prev, coords_current):
#     """
#     Computes the affine transformation including rotation and scaling.
#     """
#     # At least three points needed to calculate affine transformation
#     if coords_prev.shape[0] < 3 or coords_current.shape[0] < 3:
#         raise ValueError("Need at least three points in each frame to compute affine transformation.")

#     # Calculate affine transformation matrix
#     M = calculate_affine_matrix(coords_prev[:3], coords_current[:3])

#     # Compute rotation angle
#     angle = np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi

#     # Compute scale
#     scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
#     scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)

#     # Update the translation part of the matrix
#     M[:, 2] = coords_current.mean(axis=0) - np.dot(M[:, :2], coords_prev.mean(axis=0))

#     return M, angle, (scale_x, scale_y)

# # Load an image
# image = cv2.imread('crocs1/product_crcos_main.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Load coordinates
# coords = np.load("crocs1/coords.npy")

# # Apply transformations based on coordinates
# images = [image]  # Start with the initial image
# for i in range(1, len(coords)):
#     M, angle, scale = compute_zoom_and_rotation(coords[i-1], coords[i])
#     # print("Transformation matrix: \n", M)
#     transformed_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#     images.append(transformed_img)

# # Create video from images using moviepy
# clip = mpy.ImageSequenceClip([img[:, :, :] for img in images], fps=6)
# clip.write_videofile("project_crocs_rotated.mp4", codec='libx264')



import numpy as np
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy

def calculate_affine_matrix(pts1, pts2):
    """
    Calculate an affine transformation matrix from pts1 to pts2.
    """
    if pts1.shape[0] == 3 and pts2.shape[0] == 3:  # Exactly three points
        return cv2.getAffineTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    else:
        # Compute the affine transformation with more points using least squares
        return cv2.estimateAffinePartial2D(pts1.astype(np.float32), pts2.astype(np.float32))[0]

def apply_transform(img, M):
    """
    Applies the affine transformation to the image.
    """
    height, width = img.shape[:2]
    return cv2.warpAffine(img, M, (width, height))

def compute_zoom_and_rotation(coords_prev, coords_current):
    """
    Computes the affine transformation including rotation and scaling.
    """
    # At least three points needed to calculate affine transformation
    if coords_prev.shape[0] < 3 or coords_current.shape[0] < 3:
        raise ValueError("Need at least three points in each frame to compute affine transformation.")

    # Calculate affine transformation matrix
    M = calculate_affine_matrix(coords_prev[:3], coords_current[:3])

    # Compute rotation angle
    angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

    # Compute scale
    scale_x = np.linalg.norm(M[:, 0])
    scale_y = np.linalg.norm(M[:, 1])

    # Update the translation part of the matrix
    M[:, 2] = coords_current.mean(axis=0) - np.dot(M[:, :2], coords_prev.mean(axis=0))

    return M, angle, (scale_x, scale_y)

# Load an image
image = cv2.imread('crocs1/product_crcos_main.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load coordinates
coords = np.load("crocs1/coords.npy")

# Apply transformations based on coordinates
images = [image]  # Start with the initial image
for i in range(1, len(coords)):
    M, angle, scale = compute_zoom_and_rotation(coords[0], coords[i])
    # print("Transformation matrix: \n", M)
    transformed_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    images.append(transformed_img)

# Create video from images using moviepy
clip = mpy.ImageSequenceClip([img[:, :, :] for img in images], fps=6)
clip.write_videofile("project_crocs_rotated.mp4", codec='libx264')
