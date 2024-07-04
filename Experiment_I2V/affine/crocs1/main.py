import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from create_triangle import find_triangle
import cv2 


img = Image.open('/Users/ameerazam/Documents/affine/crocs1/crocs_mask_1.png')
coords_traigle = find_triangle(img)
print(coords_traigle)
src = cv.imread("/Users/ameerazam/Documents/affine/crocs1/product_crcos_main.png")




new_srcTri = np.array([[coords_traigle["x1"], coords_traigle["y1"]], [coords_traigle["x2"], coords_traigle["y2"]], [coords_traigle["x3"], coords_traigle["y3"]]]).astype(np.float32)
srcTri = new_srcTri
 

print("srcTri",srcTri)

# srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)


dstTri = np.array([[249.0, 131.0], [433.0, 557.0],[515.0, 300.]]).astype(np.float32)

warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

# Rotating the image after Warp

center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
angle = -50
scale = 0.6
rot_mat = cv.getRotationMatrix2D( center, angle, scale )
warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))


# print("warp_rotate_dst shape", warp_rotate_dst.shape)



#draw circle on src
src = cv2.circle(src, (coords_traigle["x1"], coords_traigle["y1"]), 5, (255, 0, 255), -1)
src = cv2.circle(src, (coords_traigle["x2"], coords_traigle["y2"]), 5, (255, 0, 255), -1)
src = cv2.circle(src, (coords_traigle["x3"], coords_traigle["y3"]), 5, (255, 0, 255), -1)




#draw circle on warp_dst   
for i in range(3):
    warp_dst = cv2.circle(warp_dst, (int(dstTri[i][0]),int( dstTri[i][1])), 5, (255, 0, 255), -1)



#draw on warp_rotate_dst    
# warp_rotate_dst = cv2.circle(warp_rotate_dst, (250,130), 5, (255, 0, 255), -1)
# warp_rotate_dst = cv2.circle(warp_rotate_dst, (402,500), 5, (255, 0, 255), -1)
# warp_rotate_dst = cv2.circle(warp_rotate_dst, (600,457), 5, (255, 0, 255), -1)



plt.subplot(121)
plt.imshow(src)
plt.title('Source Image')
plt.subplot(122)
plt.imshow(warp_dst)
plt.title('Warp and Rotate')
plt.show()
# plt.imshow(warp_rotate_dst)
# plt.title('Warp and Rotate')
# plt.show()