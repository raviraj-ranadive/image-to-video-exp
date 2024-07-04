import numpy as np

# Load coordinates
coords = np.load("/Users/ameerazam/Documents/affine/coords.npy")

#find this point
"""
Point 1: [428 270]
Point 2: [502 486]
Point 3: [544 225]
"""
Point1 =  [428 , 270]
Point2 = [502 , 486]
Point3 =  [544 , 225]

# Define the point to find or find the closest to
point_to_find = np.array(Point3)

first_coord  = coords[0]

print(first_coord.shape)

# Find the closest point to all the points in the  first_coord (1024,2)
for  indx,point_to_find in enumerate([Point1,Point2,Point3]):
    closest_point = first_coord[np.argmin(np.linalg.norm(first_coord - point_to_find, axis=1))]
    
    point_index = np.where((first_coord == closest_point).all(axis=1))
    print("Point ",indx+1,":",closest_point,"Index:",point_index)
