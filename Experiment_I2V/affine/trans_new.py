import cv2 as cv
import numpy as np
from PIL import Image
import moviepy.editor as mpy
import os

def apply_transform(src, srcTri, dstTri, angle=-50, scale=0.6):
    """ Apply affine and rotation transformations to the source image. """
    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

    center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

    return warp_dst, warp_rotate_dst




src = cv.imread("/Users/ameerazam/Documents/affine/product.png",cv.IMREAD_UNCHANGED)
coords  = np.load("/Users/ameerazam/Documents/affine/coords.npy")
video = cv.VideoCapture("/Users/ameerazam/Documents/affine/video.mp4")

mask = os.listdir("/Users/ameerazam/Downloads/mask")
#sort the mask name as 1, 2,3
mask.sort(key=lambda x: int(x.split(".")[0]))


all_masks = []
for msk in mask:
    msk = cv.imread(f"/Users/ameerazam/Downloads/mask/{msk}",cv.IMREAD_GRAYSCALE)
    all_masks.append(msk)


video_frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    video_frames.append(frame)

# Initialize transformation points
pts1, pts2, pts3 = 298, 684, 205
srcTri = np.array([[coords[0][pts1][0], coords[0][pts1][1]],
                   [coords[0][pts2][0], coords[0][pts2][1]],
                   [coords[0][pts3][0], coords[0][pts3][1]]], dtype=np.float32)



# Process each frame


frames = [] 
if True:
    for index in range(1, len(coords)):
        dstTri = np.array([[coords[index][pts1][0], coords[index][pts1][1]],
                        [coords[index][pts2][0], coords[index][pts2][1]],
                        [coords[index][pts3][0], coords[index][pts3][1]]], dtype=np.float32)
        
        warp_dst, _ = apply_transform(src, srcTri, dstTri)
        
        # Convert to PIL image for blending
        warp_dst = cv.cvtColor(warp_dst, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(warp_dst).convert('RGBA')
        video_pil = Image.fromarray(video_frames[index]).convert('RGBA')
        final = Image.blend(video_pil, pil_image, 0.75)
        frames.append(np.array(final))



st_index =0



final_product_images = []

for index in range(len(coords)-1):
    bin_mask = cv.threshold(all_masks[index], 127, 1, cv.THRESH_BINARY)[1]
    #multi with frames 
    rgba_frames = cv.bitwise_and(frames[index], frames[index], mask=bin_mask) 
    final_product_images.append(rgba_frames)
    # break

Image.fromarray(final_product_images[0]).save("test.png")

# for index in range(len(frames)):
#     #past frame on video frame with opacity 0.8 using pillow 
#     video_frame = Image.fromarray(video_frames[index])
#     video_frame = video_frame.convert('RGBA')
#     frame = Image.fromarray(frames[index])
#     frame = frame.convert('RGBA')
#     final = Image.blend(video_frame, frame, 0.75)
#     final_product_images.append(np.array(final))
#     # break


# Create and save the video file
clip = mpy.ImageSequenceClip(final_product_images, fps=6)
clip.write_videofile('trans_new_mask.mp4')
