import numpy as np
import cv2
import os
import cv2 as cv 



width,height = 1024, 576

# Function to apply transformation based on srcTri and dstTri
def apply_transformation(image, srcTri, dstTri):
    # Compute the affine transformation matrix
    # M = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    M = cv2.getPerspectiveTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the affine transformation to the image
    transformed_image = cv2.warpAffine(image, M, (width, height))
    
    return transformed_image



image = cv2.imread('/Users/ameerazam/Documents/affine/product.png',cv2.IMREAD_UNCHANGED)
points = np.load('/Users/ameerazam/Documents/affine/coords.npy') 


pts1 = 298 #10
pts2  = 684  #200
pts3  =  205  #720

srcTri = np.array([[points[0][pts1][0], points[0][pts1][1]],
                     [points[0][pts2][0], points[0][pts2][1]],
                     [points[0][pts3][0], points[0][pts3][1]]], dtype=np.float32)


final_frame = []
for index in (range(1,len(points)-1)):
    dstTri = np.array([[points[index][pts1][0], points[index][pts1][1]],
                        [points[index][pts2][0], points[index][pts2][1]],
                        [points[index][pts3][0], points[index][pts3][1]]], dtype=np.float32)

    transformed_image = apply_transformation(image, srcTri, dstTri)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        transformed_image = cv2.circle(transformed_image, (int(dstTri[i][0]),int( dstTri[i][1])), 5, (255, 0,0), -1)

    final_frame.append(transformed_image)


mask_files = os.listdir("/Users/ameerazam/Downloads/mask")
mask_files.sort(key=lambda x: int(x.split(".")[0]))



all_masks = [cv.imread(f"/Users/ameerazam/Downloads/mask/{m}", cv.IMREAD_GRAYSCALE) for m in mask_files]

mask_cut = [] 
for index in range(1,len(points)-1):
    mask = all_masks[index]
    mask = cv2.resize(mask, (width, height))
    mask_cut.append(mask)
    







import moviepy.editor as mpy

clip = mpy.ImageSequenceClip(final_frame, fps=6)
clip.write_videofile('final_frame.mp4')