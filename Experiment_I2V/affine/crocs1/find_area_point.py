import cv2
import numpy as np

def find_largest_triangle(contour):
    """Given a contour, returns the largest triangle in terms of area."""
    max_area = 0
    best_triangle = None

    # Check combinations of three points
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            for k in range(j + 1, len(contour)):
                # Calculate area of the triangle using the determinant method (Shoelace formula)
                pt1 = contour[i][0]
                pt2 = contour[j][0]
                pt3 = contour[k][0]
                area = abs(pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1])) / 2.0
                if area > max_area:
                    max_area = area
                    best_triangle = (pt1, pt2, pt3)

    return best_triangle

def main():
    # Load the image
    img = cv2.imread('/Users/ameerazam/Downloads/mask/1.png', cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the one we're interested in
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the largest triangle
    triangle = find_largest_triangle(largest_contour)

    #draw the point on the image
    for i in range(3):
        cv2.circle(img, (triangle[i][0], triangle[i][1]), 5, (0, 0, 0), -1)

    #  save image 
    cv2.imwrite("triangle.png", img)


    if triangle:
        print("The largest triangle points are:")
        print(f"Point 1: {triangle[0]}")
        print(f"Point 2: {triangle[1]}")
        print(f"Point 3: {triangle[2]}")
    else:
        print("No triangle found.")

if __name__ == "__main__":
    main()
