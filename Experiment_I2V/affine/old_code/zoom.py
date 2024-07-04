import cv2
import numpy as np
import matplotlib.pyplot as plt

def zoom_image(img, center, scale):
    """
    Zooms in/out on the image based on the center and scale.
    - img: input image
    - center: tuple (x, y) indicating the center of zoom
    - scale: float, >1 for zoom-in, <1 for zoom-out
    """
    height, width = img.shape[:2]

    # Translation matrix to move the center to the origin
    M_translate = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ])

    # Scaling matrix
    M_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

    # Reverse translation matrix
    M_inv_translate = np.array([
        [1, 0, center[0]],
        [0, 1, center[1]],
        [0, 0, 1]
    ])

    # Combined transformation matrix
    M = M_inv_translate @ M_scale @ M_translate
    M = M[:2, :]  # Convert to 2x3 matrix as required by warpAffine

    # Apply affine transformation
    zoomed_img = cv2.warpAffine(img, M, (width, height))
    return zoomed_img

# Load image
image = cv2.imread('product.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Zoom parameters
center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
print(center)
scale = 1.5  # Scale factor > 1 for zoom-in, < 1 for zoom-out

# Apply zoom

zoomed_in = zoom_image(image, center, scale)
scale = 0.75  # Example scale for zoom-out

zoomed_out = zoom_image(image, center, scale)

# Display images
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(132)
plt.imshow(zoomed_in)
plt.title("Zoomed In")
plt.axis('off')

plt.subplot(133)
plt.imshow(zoomed_out)
plt.title("Zoomed Out")
plt.axis('off')

plt.show()
