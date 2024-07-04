from PIL import Image
import numpy as np

# Load the image



def find_triangle(img):
    # Convert the image to grayscale and then to a numpy array
    img_gray = img.convert('L')
    img_array = np.array(img_gray)

    # Threshold the image to detect the white parts
    threshold = 250
    mask = img_array > threshold

    # Find the coordinates of the white parts
    coordinates = np.argwhere(mask)

    # Find the top, bottom, and rightmost points to form a triangle
    top = coordinates[coordinates[:,0].argmin()]
    bottom = coordinates[coordinates[:,0].argmax()]
    right = coordinates[coordinates[:,1].argmax()]

    coords ={"x1": top[1], "y1": top[0], "x2": bottom[1], "y2": bottom[0], "x3": right[1], "y3": right[0]}
    return coords
