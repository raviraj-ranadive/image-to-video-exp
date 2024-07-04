import cv2 as cv 
import cv2
import numpy as np
import os 
import moviepy.editor as mpy

def contour_box(base_path,mask_path,use_padding,padding_pixel,padding_box_product):

    mask = os.listdir(mask_path)
    all_data_point = []
    mask.sort(key=lambda x: int(x.split(".")[0]))
    all_masks = []
    for msk in mask:
        msk = cv.imread(f"{mask_path}/{msk}",cv.IMREAD_GRAYSCALE)
        msk =cv2.resize(msk,(1024,576))
        if use_padding:
            msk = cv2.copyMakeBorder(msk, padding_pixel, padding_pixel, padding_pixel, padding_pixel, cv2.BORDER_CONSTANT, value=0)
        # msk= cv.resize(msk,(3000,3000))
        all_masks.append(msk)

    frames_mask = []
    for idx,mask in enumerate(all_masks):
        image = mask#cv2.imread('/Users/ameerazam/Documents/affine/crocs2/mask.png', cv2.IMREAD_GRAYSCALE)
        # Threshold the image to isolate the mask
        _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assume the largest contour is the mask
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Top and bottom coordinates of the bounding box
        top_edge = [(x, y), (x + w, y)]
        bottom_edge = [(x, y + h), (x + w, y + h)]

        #padding 
        padding = padding_box_product
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        top_edge = [(x, y), (x + w, y)]
        bottom_edge = [(x, y + h), (x + w, y + h)]

        #drwa the bounding box on the image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #draw point on the image
        cv2.circle(image, top_edge[0], 5, (0, 0, 255), -1)
        cv2.circle(image, top_edge[1], 5, (0, 0, 255), -1)
        cv2.circle(image, bottom_edge[0], 5, (0, 0, 255), -1)
        cv2.circle(image, bottom_edge[1], 5, (0, 0, 255), -1)

        #print all the coordinates of corners
        if False:
            print("Top Left: ", top_edge[0])
            print("Top Right: ", top_edge[1])
            print("Bottom Left: ", bottom_edge[0])
            print("Bottom Right: ", bottom_edge[1])
        os.makedirs(f"{base_path}/boxed_mask", exist_ok=True)
        cv2.imwrite(f"{base_path}/boxed_mask/{idx}.png", image)
        frames_mask.append(image)
        all_data_point.append([top_edge[0],top_edge[1],bottom_edge[0],bottom_edge[1]])

    np.save(f"{base_path}/mask_coords.npy",all_data_point)
    return frames_mask




"""
Dirs --  Data we need till stage 

Input must be  need this stage:
--/mask
--/O.png which is 0th mask
--/coords.npy which is from cotrack for future need in case 
--/interpolated.mp4 from FILM interpolation
--/porduct.png 
--/video.mp4 which orignal vidoe before FILM
Return:
--/mask_coords.npy Using all --/mask 
--/boxed_mask/ will create new dirs for show all  mask contour Areas
--/frames_mask_.mp4

"""
from driven_using_new import past_product_main

for ind in [10,11,12,13,14,15]:
    base_path = f"/Users/ameerazam/Documents/affine/i2v-test-new/test-{ind}"
    mask_path = f"{base_path}/mask"
    use_padding = True
    padding_pixel = 1000
    padding_box_product = 0
    frames_mask = contour_box(base_path,mask_path,use_padding,padding_pixel,padding_box_product)



    past_product_main(base_path,use_padding,padding_pixel)


    clip = mpy.ImageSequenceClip(frames_mask, fps=25)
    clip.write_videofile(f'{base_path}/frames_mask_.mp4')



