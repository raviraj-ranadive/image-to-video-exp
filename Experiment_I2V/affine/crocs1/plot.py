# import numpy as np 

# import cv2 
# import moviepy.editor as mpy

# coords  = np.load("./coords.npy")
# coords = coords - 100

# #dummy white image  w,h 1024, 576



# frames  =  []

# for coord in coords:
#     pts1= (250,150)
#     pts2 = (700,140)
#     pts3 = (513,506)
#     print("coord",coord)
#     print("coord",coord[pts1[0]])
#     dummy = np.ones((576,1024,3),np.uint8)*255
#     #pint is 200,250,720
#     image = cv2.circle(dummy, (coord[200]), 5, (255, 0, 0), -1)
#     image = cv2.circle(image, (coord[250]), 5, (255, 255, 0), -1)
#     image = cv2.circle(image, (coord[720]), 5, (255, 0, 255), -1)
    
#     frames.append(image)


#     # break 

# clip = mpy.ImageSequenceClip(frames, fps=6)
# #in mp4 
# clip.write_videofile('output.mp4')


import numpy as np
import cv2  as cv
import os 
from PIL import Image


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
    video_frames.append(frame)

video_frames_copy = video_frames.copy()


print("video_frames ",len(video_frames))
print("all_masks ",len(all_masks))


final_frames =[]
for i in range(len(all_masks)):
    #convert mask in 0,1 
    erode = cv.erode(all_masks[i], np.ones((5,5),np.uint8), iterations=1)
    bin_mask = cv.threshold(erode, 127, 1, cv.THRESH_BINARY)[1]
    # bin_mask = cv.threshold(all_masks[i], 127, 1, cv.THRESH_BINARY)[1]
    #multiply mask with video frame
    video_frames[i] = cv.bitwise_and(video_frames[i], video_frames[i], mask=bin_mask)
    #convert to pil because moviepy works with pil
    video_frames[i] = cv.cvtColor(video_frames[i], cv.COLOR_BGR2RGB)
    final_frames.append(video_frames[i])





#past final_frames on   video_frames
final = []
for i in range(len(final_frames)):
    final_frame = Image.fromarray(final_frames[i]).convert("RGBA")
    video_frame = Image.fromarray(video_frames_copy[i]).convert("RGBA")
    mask_frame = Image.fromarray(all_masks[i]).convert("L")

    #apply mask as forgeground
    video_frame.paste(mask_frame, (0, 0), mask_frame)
    #apply alpha_composite


    final_frame = Image.alpha_composite(video_frame, final_frame)
    final_frame = cv.cvtColor(np.array(final_frame), cv.COLOR_RGBA2RGB)
    final.append(final_frame)





import moviepy.editor as mpy
clip = mpy.ImageSequenceClip(final, fps=6)
#in mp4
clip.write_videofile('final.mp4')



# import moviepy.editor as mpy
# clip = mpy.ImageSequenceClip(final_frames, fps=6)
# #in mp4
# clip.write_videofile('binary-test.mp4')


# #make all_masks video 
# all_masks = [cv.cvtColor(mask, cv.COLOR_GRAY2BGR) for mask in all_masks]
# clip = mpy.ImageSequenceClip(all_masks, fps=6)
# #in mp4
# clip.write_videofile('binary-mask.mp4')


