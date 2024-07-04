import cv2 as cv
import numpy as np
from PIL import Image
from create_triangle import find_triangle
import cv2 
import os 


def apply_transform(src,srcTri,dstTri):    

    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

    
    return warp_dst


src = cv.imread("/Users/ameerazam/Documents/affine/product.png",cv.IMREAD_UNCHANGED)
coords  = np.load("/Users/ameerazam/Documents/affine/coords.npy")
coord1s = coords
video = cv2.VideoCapture("/Users/ameerazam/Documents/affine/video.mp4")

mask = os.listdir("/Users/ameerazam/Downloads/mask")
#sort the mask name as 1, 2,3
mask.sort(key=lambda x: int(x.split(".")[0]))


all_mask = []
for msk in mask:
    msk = cv.imread(f"/Users/ameerazam/Downloads/mask/{msk}",cv.IMREAD_GRAYSCALE)
    all_mask.append(msk)


video_frame = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    video_frame.append(frame)
    # break
frames = []
st_index =0
pts1 = 298 #10
pts2  = 684  #200
pts3  =  205  #720
# coord1 = coords[0][pts1]
# coord2 = coords[0][pts2]
# coord3 = coords[0][pts3]


# srcTri = np.array([[coord1[0],coord1[1]],[coord2[0],coord2[1]],[coord3[0],coord3[1]]]).astype(np.float32)


for index in range(1,len(coords)):
    coord1 = coords[index-1][pts1]
    coord2 = coords[index-1][pts2]
    coord3 = coords[index-1][pts3]
    
    srcTri = np.array([[coord1[0],coord1[1]],[coord2[0],coord2[1]],[coord3[0],coord3[1]]]).astype(np.float32)
    
    coord1 = coords[index][pts1]
    coord2 = coords[index][pts2]
    coord3 = coords[index][pts3]

    dstTri = np.array([[coord1[0],coord1[1]],[coord2[0],coord2[1]],[coord3[0],coord3[1]]]).astype(np.float32)
    delta = dstTri - srcTri

    #add in dstTri
    dstTri = srcTri + delta

    
    warp_dst = apply_transform(src,srcTri,dstTri)
    #cv2 to pil because moviepy works with pil
    #draw point on warp_dst
    for i in range(3):
        warp_dst = cv.circle(warp_dst, (int(dstTri[i][0]),int( dstTri[i][1])), 5, (255, 0,0), -1)

    warp_dst = cv.cvtColor(warp_dst, cv.COLOR_BGR2RGB)
    warp_dst = Image.fromarray(warp_dst)
    warp_dst = warp_dst.convert('RGBA')
    frames.append(np.array(warp_dst))


video_frames = video_frame[st_index:st_index+len(frames)]
all_masks = all_mask[st_index:st_index+len(frames)]

final_product_images = []

# for index in range(len(frames)):
#     bin_mask = cv.threshold(all_masks[index], 127, 1, cv.THRESH_BINARY)[1]
#     #multi with frames 
#     rgba_frames = cv.bitwise_and(frames[index], frames[index], mask=bin_mask) 

#     final_product_images.append(rgba_frames)
#     # break

for index in range(len(frames)):
    #past frame on video frame with opacity 0.8 using pillow 
    video_frame = Image.fromarray(video_frames[index])
    video_frame = video_frame.convert('RGBA')
    frame = Image.fromarray(frames[index])
    frame = frame.convert('RGBA')
    final = Image.blend(video_frame, frame, 0.8)
    final_product_images.append(np.array(final))
    # break


    





import moviepy.editor as mpy
clip = mpy.ImageSequenceClip(final_product_images, fps=6)
#in mp4
clip.write_videofile('delta_with_st_dst.mp4')



clip = mpy.ImageSequenceClip(frames, fps=6)
#in mp4
clip.write_videofile('final_product_images.mp4')


