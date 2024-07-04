# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# def scale_image(image_path, fx, fy, method=cv.INTER_CUBIC):
#     img = cv.imread(image_path)
#     assert img is not None, "File could not be read, check with os.path.exists()"
#     resized = cv.resize(img, None, fx=fx, fy=fy, interpolation=method)
#     return resized

# def translate_image(image_path, dx, dy):
#     img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     assert img is not None, "File could not be read, check with os.path.exists()"
#     rows, cols = img.shape
#     M = np.float32([[1, 0, dx], [0, 1, dy]])
#     translated = cv.warpAffine(img, M, (cols, rows))
#     return translated

# def rotate_image(image_path, angle, scale=1.0):
#     img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#     assert img is not None, "File could not be read, check with os.path.exists()"
#     rows, cols = img.shape
#     center = ((cols - 1) / 2.0, (rows - 1) / 2.0)
#     M = cv.getRotationMatrix2D(center, angle, scale)
#     rotated = cv.warpAffine(img, M, (cols, rows))
#     return rotated

# def affine_transform(image_path, pts1, pts2):
#     img = cv.imread(image_path)
#     assert img is not None, "File could not be read, check with os.path.exists()"
#     rows, cols, ch = img.shape
#     M = cv.getAffineTransform(pts1, pts2)
#     transformed = cv.warpAffine(img, M, (cols, rows))
#     return transformed

# def perspective_transform(image_path, pts1, pts2):
#     img = cv.imread(image_path)
#     assert img is not None, "File could not be read, check with os.path.exists()"
#     rows, cols, ch = img.shape
#     M = cv.getPerspectiveTransform(pts1, pts2)
#     warped = cv.warpPerspective(img, M, (300, 300))
#     return warped

# # Main execution block to test the functions
# if __name__ == "__main__":
#     img_path = "/Users/ameerazam/Documents/affine/product_white.png"
#     scaled_img = scale_image(img_path, 2, 2)
#     translated_img = translate_image(img_path, 100, 50)
#     rotated_img = rotate_image(img_path, 90)
#     affine_img = affine_transform(img_path, np.float32([[50, 50], [200, 50], [50, 200]]), np.float32([[10, 100], [200, 50], [100, 250]]))
#     perspective_img = perspective_transform(img_path, np.float32([[56, 65], [368, 52], [28, 387], [389, 390]]), np.float32([[0, 0], [300, 0], [0, 300], [300, 300]]))

#     # Show images
#     plt.figure(figsize=(10, 8))
#     plt.subplot(231), plt.imshow(cv.cvtColor(scaled_img, cv.COLOR_BGR2RGB)), plt.title('Scaled')
#     plt.subplot(232), plt.imshow(cv.cvtColor(translated_img, cv.COLOR_BGR2RGB)), plt.title('Translated')
#     plt.subplot(233), plt.imshow(cv.cvtColor(rotated_img, cv.COLOR_BGR2RGB)), plt.title('Rotated')
#     plt.subplot(234), plt.imshow(cv.cvtColor(affine_img, cv.COLOR_BGR2RGB)), plt.title('Affine')
#     plt.subplot(235), plt.imshow(cv.cvtColor(perspective_img, cv.COLOR_BGR2RGB)), plt.title('Perspective')
#     plt.show()



import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def apply_affine_transform(img, pts1, pts2):
    M = cv.getAffineTransform(pts1, pts2)
    transformed_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return transformed_img

def apply_perspective_transform(img, pts1, pts2):
    M = cv.getPerspectiveTransform(pts1, pts2)
    transformed_img = cv.warpPerspective(img, M, (300, 300))  # You may adjust the size as necessary
    return transformed_img

# Load image
img_path = "/Users/ameerazam/Documents/affine/product.png"
coords = np.load("/Users/ameerazam/Documents/affine/coords.npy")
base_image = cv.imread(img_path)
assert base_image is not None, "File could not be read, check with os.path.exists()"

