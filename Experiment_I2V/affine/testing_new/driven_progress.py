import numpy as np
import cv2
from PIL import Image
import moviepy.editor as mpy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import statsmodels.api as sm
from scipy.stats import norm

# Function to compute Exponential Moving Average
def exponential_moving_average(new_value, average, alpha=0.2):
    return alpha * new_value + (1 - alpha) * average


def remove_padding(vid_frame, padding_pixel):
        height, width = vid_frame.shape[:2]  # get height and width
        return vid_frame[padding_pixel:height-padding_pixel, padding_pixel:width-padding_pixel]

def find_all_triangles(points):
        from itertools import combinations
        return [np.float32(triangle) for triangle in combinations(points, 3)]


def resize_image_with_scale(image, scale):
    # Get the original dimensions
    original_width, original_height = image.size
    
    # Calculate the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def double_exponential_smoothing(X, alpha, beta):
    # Initialize output array with the same shape as X
    S = np.zeros(X.shape)
    
    # Iterate over the second and third dimensions
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            # Extract the 1D time series
            X_1d = X[:, i, j]
            
            # Initialize arrays A, B for this 1D series
            A, B = (np.zeros(X_1d.shape[0]) for _ in range(2))
            
            # Initial conditions
            A[0] = X_1d[0]
            B[0] = X_1d[1] - X_1d[0]
            
            # Apply smoothing
            for t in range(1, X_1d.shape[0]):
                A[t] = alpha * X_1d[t] + (1 - alpha) * (A[t - 1] + B[t - 1])
                B[t] = beta * (A[t] - A[t - 1]) + (1 - beta) * B[t - 1]
                S[t, i, j] = A[t] + B[t]

    return S


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# Design a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def past_product_main(base_path,use_padding,padding_pixel):

    finals_new =[]
    product_image = cv2.imread(f'{base_path}/product.png', cv2.IMREAD_UNCHANGED)
    product_image =cv2.resize(product_image,(1024,576))
    if use_padding:
        product_image = cv2.copyMakeBorder(product_image, padding_pixel, padding_pixel, padding_pixel, padding_pixel, cv2.BORDER_CONSTANT, value=0)
    product_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2RGBA)


    video_name_file = "interpolated"
    video = cv2.VideoCapture(f'{base_path}/{video_name_file}.mp4')
    video_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame =cv2.resize(frame,(1024,576))
        if use_padding:
            frame = cv2.copyMakeBorder(frame, padding_pixel, padding_pixel, padding_pixel, padding_pixel, cv2.BORDER_CONSTANT, value=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames.append(frame)

    video.release()

    mask_coords = np.load(f"{base_path}/mask_coords.npy")
    mask_rect = cv2.imread(f'{base_path}/boxed_mask/0.png', cv2.IMREAD_UNCHANGED)
    (h, w) = mask_rect.shape[:2]

    index =1
    coord1, coord2, coord3, coord4 = mask_coords[index-1]
    src_triangles = find_all_triangles(np.float32([coord1, coord2, coord3, coord4 ])) #find_centroid([coord1, coord2, coord3, coord4])

    total_average  = []
    exp_average = []
    previous_average_matrix = None
    alpha = 0.3 # Smoothing factor for EMA


    total_average  = []
    expo__average = []
    
    finals = []

    for index in range(1, len(mask_coords)):
        pts11 = mask_coords[index][0]
        pts22 = mask_coords[index][1]
        pts33 = mask_coords[index][2]
        pts44 = mask_coords[index][3]


        dst_triangles = find_all_triangles(np.float32([pts11, pts22, pts33, pts44, ]))#find_centroid([pts11, pts22, pts33, pts44])

        accumulated_matrix = np.zeros((2, 3), dtype=np.float32)
        all_M = []

        for src_triangle, dst_triangle in zip(src_triangles, dst_triangles):
            transformation_matrix = cv2.getAffineTransform(src_triangle, dst_triangle)
            accumulated_matrix += transformation_matrix
            all_M.append(transformation_matrix)

        average_matrix = np.mean(np.array(all_M), axis=0)

        total_average.append(average_matrix)
        if previous_average_matrix is not None:
            average_matrix = exponential_moving_average(average_matrix, previous_average_matrix, alpha) 
            exp_average.append(previous_average_matrix)

        previous_average_matrix = average_matrix
        expo__average.append(previous_average_matrix)

        warped_image = cv2.warpAffine(product_image, average_matrix, (w, h), flags=cv2.INTER_CUBIC)
        finals.append(warped_image)

    # filtered_matrices =  apply_spline_interpolation(expo__average)

    # Apply the Savitzky-Golay filter to each element (i, j) across all matrices
    filtered_matrices_arr  =  np.array(total_average)

    # Dictionary to store the arrays
    
    filtered_total_butter_low_pass = np.zeros_like(total_average)
    dict_prev = {}
    dict_next = {}
    for i in range(2):
        for j in range(3):
           
            element_array = filtered_matrices_arr[:, i, j]

            order = 2
            fs = 30.0       # sample rate, Hz
            cutoff = 0.65 # desired cutoff frequency of the filter, Hz

            # Apply the filter
            filtered_data = butter_lowpass_filter(element_array, cutoff, fs, order)
            dict_next[f"{i}{j}"] = filtered_data
            dict_prev[f"{i}{j}"] = element_array
            filtered_total_butter_low_pass[:, i, j] = filtered_data


    plt.figure(figsize=(15, 10))
    for key, values in dict_prev.items():
        plt.plot(values, label=f'Mat i-j Index {key}')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Graphs for each Key in the Dictionary')
    plt.legend()
    plt.grid(True)
    # Save the combined figure with maxima markers
    dict_plot  = f'{base_path}/dict_prev.png'
    plt.tight_layout()
    plt.savefig(dict_plot)

    plt.figure(figsize=(15, 10))
    for key, values in dict_next.items():
        plt.plot(values, label=f'Mat i-j Index {key}')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Graphs for each Key in the Dictionary')
    plt.legend()
    plt.grid(True)
    # Save the combined figure with maxima markers
    dict_plot  = f'{base_path}/dict_next.png'
    plt.tight_layout()
    plt.savefig(dict_plot)

    filtered_matrices = [filtered_total_butter_low_pass[k] for k in range(filtered_total_butter_low_pass.shape[0])]

    for index in range(0, len(filtered_matrices)-1):
        average_matrix_ = filtered_matrices[index]
        warped_image = cv2.warpAffine(product_image, average_matrix_, (w, h), flags=cv2.INTER_CUBIC)
        finals_new.append(warped_image)
    
    
    # Paste final images on video frames and save
    pasted_video = []
    pasted_video_ = []
    scale = 1.5  # Example scale factor

    # print("finals_new len ", len(finals_new)-1)
    for i in range(len(finals_new)-1):
        vid_frame = Image.fromarray(video_frames[i]).convert('RGBA')
        img_frame = Image.fromarray(finals[i]).convert('RGBA')

        vid_frame_ = Image.fromarray(video_frames[i]).convert('RGBA')
        img_frame_ = Image.fromarray(finals_new[i]).convert('RGBA')

        # img_frame = resize_image_with_scale(img_frame, scale)
        # vid_frame = resize_image_with_scale(vid_frame, scale)

        vid_frame.paste(img_frame, (0, 0), img_frame)
        vid_frame_.paste(img_frame_, (0, 0), img_frame_)


        if use_padding:
            vid_frame_cropped_= remove_padding(np.array(vid_frame_),padding_pixel)
        else:
            vid_frame_cropped_ = np.array(vid_frame_)
            

        pasted_video_.append(vid_frame_cropped_)


        if use_padding:
            vid_frame_cropped = remove_padding(np.array(vid_frame),padding_pixel)
        else:
            vid_frame_cropped = np.array(vid_frame)
             

        pasted_video.append(vid_frame_cropped)

    clip = mpy.ImageSequenceClip(pasted_video, fps=40)
    clip.write_videofile(f'{base_path}/Video_exp.mp4')
    clip = mpy.ImageSequenceClip(pasted_video_, fps=40)
    clip.write_videofile(f'{base_path}/video_butter_low_pass.mp4')






"""
Dirs --  Data need in this:

Input must be  need this stage:
--/mask_coords.npy Using all --/mask
--/porduct.png  with RGBA trabsparent Image
--/interpolated.mp4 from FILM interpolation
--/coords.npy in case there but not using in this process
--/video.mp4 in case there but not using in this process

Return:
--/driven.mp4 return will be a video after paster on driven video

"""
# for ind in [i for i in range(4,11)]:
ind  =  15

base_path = f"/Users/ameerazam/Documents/affine/i2v-test-new/test-{ind}"

mask_path = f"{base_path}/mask"
use_padding = True
padding_pixel = 1000
past_product_main(base_path,use_padding,padding_pixel)


exit()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import norm
from scipy.fft import fft, ifft

# Step 1: Define the data
dict_i_j = {}
for i in range(2):
    for j in range(3):
        element_array = np.load(f"{base_path}/element_array_{i}_{j}.npy")
        dict_i_j[f"{i}{j}"] = element_array


# np.save(f"{base_path}/modified_data.npy", modified_data)
plt.figure(figsize=(15, 10))
for key, values in dict_i_j.items():
    plt.plot(values, label=f'Mat i-j Index {key}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Graphs for each Key in the Dictionary')
plt.legend()
plt.grid(True)
# Save the combined figure with maxima markers
dict_plot  = f'{base_path}/dict_plot.png'
plt.tight_layout()
plt.savefig(dict_plot)


if False:
    for key in dict_i_j:
        # Step 2: Compute Fourier Transform
        data = dict_i_j[key]
        from scipy.signal import butter, filtfilt

        # Design a Butterworth low-pass filter
        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        # Filter requirements.
        order = 2
        fs = 30.0       # sample rate, Hz
        cutoff = 0.7 # desired cutoff frequency of the filter, Hz

        # Apply the filter
        filtered_data = butter_lowpass_filter(data, cutoff, fs, order)
        dict_i_j[key] = filtered_data


        # # Compute Fourier Transform
        # ft_data = fft(data)
        # frequencies = np.fft.fftfreq(len(data))
        # magnitude = np.abs(ft_data)

        # # Set up the figure for multiple plots
        # fig, axs = plt.subplots(2, 1, figsize=(14, 12))

        # # Plot Fourier Transform
        # axs[0].plot(frequencies, magnitude)
        # axs[0].set_title('Fourier Transform')
        # axs[0].set_xlabel('Frequency')
        # axs[0].set_ylabel('Magnitude')
        # axs[0].grid(True)

        # # Histogram with Distribution Curve
        # axs[1].hist(magnitude, bins=30, density=True, alpha=0.6, color='g')
        # mu, std = norm.fit(magnitude)
        # xmin, xmax = axs[1].get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)
        # axs[1].plot(x, p, 'k', linewidth=2)
        # axs[1].set_title("Frequency Magnitude Distribution\nFit results: mu = %.2f, std = %.2f" % (mu, std))
        # plt.tight_layout()

        # # Low-pass Filter: Remove frequencies above a certain threshold
        # cutoff_frequency = 0.6  # Adjust this value based on your specific needs
        # low_pass_indices = np.abs(frequencies) <= cutoff_frequency
        # filtered_ft_data = np.zeros_like(ft_data)
        # filtered_ft_data[low_pass_indices] = ft_data[low_pass_indices]

        # # Inverse Fourier Transform to get the modified array
        # modified_data = ifft(filtered_ft_data).real




        # ft_data = fft(data)
        # frequencies = np.fft.fftfreq(len(data))
        # magnitude = np.abs(ft_data)

        # # Step 3: Set up the figure for multiple plots
        # fig, axs = plt.subplots(2, 1, figsize=(14, 12))

        # # Step 4: Plot Fourier Transform
        # axs[0].plot(frequencies, magnitude)
        # axs[0].set_title('Fourier Transform')
        # axs[0].set_xlabel('Frequency')
        # axs[0].set_ylabel('Magnitude')
        # axs[0].grid(True)

        # # Step 5: Frequency Histogram with Distribution Curve
        # axs[1].hist(magnitude, bins=30, density=True, alpha=0.6, color='g')

        # # Fit a normal distribution to the data
        # mu, std = norm.fit(magnitude)

        # # Plot the distribution curve
        # xmin, xmax = axs[1].get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)
        # axs[1].plot(x, p, 'k', linewidth=2)
        # axs[1].set_title("Frequency Magnitude Distribution\nFit results: mu = %.2f, std = %.2f" % (mu, std))

        # # Show all plots
        # plt.tight_layout()
        # plt.savefig(f"{base_path}/final_ftt_hist_gau_curve.png")

        # # Identify indices where the magnitude is within mu ± 2*std
        # indices = (magnitude > (mu - 3 * std)) & (magnitude < (mu + 3 * std))

        # # Filter the Fourier Transform using these indices
        # filtered_ft_data = np.zeros_like(ft_data)
        # filtered_ft_data[indices] = ft_data[indices]

        # # Perform an inverse Fourier Transform to get the modified array
        # modified_data = ifft(filtered_ft_data).real
        # dict_i_j[key] = modified_data



# np.save(f"{base_path}/modified_data.npy", modified_data)
plt.figure(figsize=(15, 10))
for key, values in dict_i_j.items():
    plt.plot(values, label=f'Mat i-j Index {key}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Graphs for each Key in the Dictionary')
plt.legend()
plt.grid(True)
# Save the combined figure with maxima markers
dict_plot  = f'{base_path}/dict_plot_smooth.png'
plt.tight_layout()
plt.savefig(dict_plot)













# # exit()
# arra_3 = np.load("/Users/ameerazam/Documents/affine/test-27may/test-new-3/element_array_1_2.npy")
# arra_5  =np.load("/Users/ameerazam/Documents/affine/test-27may/test-new-5/element_array_0_2.npy")


# # Calculate SMA and EMA for degrees between 2-10
# degrees = range(50,51)
# sma_3 = {degree: calculate_sma(arra_3, degree) for degree in degrees}
# ema_3 = {degree: calculate_ema(arra_3, degree) for degree in degrees}
# sma_5 = {degree: calculate_sma(arra_5, degree) for degree in degrees}
# ema_5 = {degree: calculate_ema(arra_5, degree) for degree in degrees}

# # Plotting the results
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# # Plot SMA for arra_3
# for degree in degrees:
#     axs[0, 0].plot(sma_3[degree], label=f'SMA degree {degree}')
# axs[0, 0].set_title('SMA for arra_3')
# axs[0, 0].legend()

# # Plot EMA for arra_3
# for degree in degrees:
#     axs[0, 1].plot(ema_3[degree], label=f'EMA degree {degree}')
# axs[0, 1].set_title('EMA for arra_3')
# axs[0, 1].legend()

# # Plot SMA for arra_5
# for degree in degrees:
#     axs[1, 0].plot(sma_5[degree], label=f'SMA degree {degree}')
# axs[1, 0].set_title('SMA for arra_5')
# axs[1, 0].legend()

# # Plot EMA for arra_5
# for degree in degrees:
#     axs[1, 1].plot(ema_5[degree], label=f'EMA degree {degree}')
# axs[1, 1].set_title('EMA for arra_5')
# axs[1, 1].legend()

# plt.tight_layout()
# plt.show()
