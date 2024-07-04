import cv2 as cv
import numpy as np
from PIL import Image
import moviepy.editor as mpy
import os

def apply_transform(src, srcTri, dstTri):
    """ Apply affine transformation to the source image. """
    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
    return warp_dst

# Load source image and coordinates
src = cv.imread("/Users/ameerazam/Documents/affine/crocs1/product_image.png", cv.IMREAD_UNCHANGED)

coords = np.load("/Users/ameerazam/Documents/affine/coords.npy")

video = cv.VideoCapture("/Users/ameerazam/Documents/affine/video.mp4")

mask_files = os.listdir("/Users/ameerazam/Downloads/mask")

mask_files.sort(key=lambda x: int(x.split(".")[0]))
all_masks = [cv.imread(f"/Users/ameerazam/Downloads/mask/{m}", cv.IMREAD_GRAYSCALE) for m in mask_files]



# Load video frames
video_frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    video_frames.append(frame)
video.release()


# Initialize transformation points


pts1 = 298 #10
pts2  = 684  #200
pts3  =  205  #720
# pts4  =  720  #720
frames = []


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
    # print(np.asarray(delta).astype(np.int32), delta)
    dstTri = srcTri + delta


    warp_dst = apply_transform(src, srcTri, dstTri)

    # Convert to PIL image for blending why  ? - 
    warp_dst = cv.cvtColor(warp_dst, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(warp_dst).convert('RGBA')
    video_pil = Image.fromarray(video_frames[index]).convert('RGBA')
    
    frames.append(np.array(pil_image))

# Image.fromarray(frames[0]).save(f"./test/final_{0}.png")

# Apply masks and create final images
final_product_images = []
for index in range(len(coords) - 1):
    bin_mask = cv.threshold(all_masks[index], 127, 255, cv.THRESH_BINARY)[1]
    bin_mask = cv.resize(bin_mask, (frames[index].shape[1], frames[index].shape[0]))
    mask_3channel = cv.cvtColor(bin_mask, cv.COLOR_GRAY2BGR)

    rgba_frame = cv.cvtColor(frames[index], cv.COLOR_RGB2RGBA)
    rgba_mask = np.dstack([mask_3channel, bin_mask])
    masked_frame = cv.bitwise_and(rgba_frame, rgba_mask)
    final_product_images.append(masked_frame)
    # break 




#past all  final_product_images on frames list 
pasted_frames = []
for index in range(len(coords) - 1):
    frame = Image.fromarray(video_frames[index]).convert('RGBA')
    final_product_image = Image.fromarray(final_product_images[index]).convert('RGBA')

    # frame.save("./test/frame.png")
    # final_product_image.save("./test/final_product_image.png")

    l1 = np.array(frame)
    #frame is background and final_product_image is foreground
    final = Image.alpha_composite(frame, final_product_image)
    for i in range(3):  
            cv.circle(np.array(final), (int(coords[index][i][0]),int( coords[index][i][1])), 5, (255, 0,0), -1)

    l2 = np.array(final)

    #find loss 
    # loss = np.sum(np.abs(l1 - l2))
    # print("loss --->",loss)
    pasted_frames.append(np.array(final))






# Save the first frame as an image for testing
# Image.fromarray(final_product_images[15]).save("test.png")


# Create and save the video file
clip = mpy.ImageSequenceClip(frames, fps=6)
clip.write_videofile('trans__bad_frames.mp4')


# Create and save the video file
clip = mpy.ImageSequenceClip(final_product_images, fps=6)
clip.write_videofile('trans__bad_final_product_images.mp4')


# Create and save the video file
clip = mpy.ImageSequenceClip(pasted_frames, fps=6)
clip.write_videofile('trans__bad_pasted_frames.mp4')
