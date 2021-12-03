import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import glob
from scipy.signal import butter,filtfilt

# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_highpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def DFFTnp(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def reverseDFFTnp(dfft):
    f_ishift = np.fft.ifftshift(dfft)
    reverse_image = np.fft.ifft2(f_ishift)
    return reverse_image

def showDFFT(img):
    fshift = DFFTnp(img)

    w, h = fshift.shape
    fshift_1 = []
    fshift_2 = []
    for i in range(w):
        fshift_1.append(butter_highpass_filter(fshift[i], cutoff, fs, order))
        fshift_2.append(butter_lowpass_filter(fshift[i], cutoff, fs, order))
    
    maxpix = fshift[w//2][h//2]
    for i in range(w):
        for j in range(h):
            if i != w//2 and j != h//2:
                if abs(np.abs(fshift[i][j])-np.abs(maxpix)) < np.abs(maxpix) - 270000:
                    fshift[i][j] = 0

    reverse_image_1 = reverseDFFTnp(fshift_1)
    reverse_image_2 = reverseDFFTnp(fshift_2)
    reverse_image = reverseDFFTnp(fshift)

    for i in range(w):
        for j in range(h):
                reverse_image[i][j] = max(reverse_image[i][j], max(reverse_image_2[i][j], reverse_image_1[i][j]))
   
    plt.subplot(121), plt.title('Input image')
    plt.imshow(abs(img), cmap='gray')
    plt.subplot(122), plt.title('Result image')
    plt.imshow(abs(reverse_image), cmap='gray')
    plt.show()

folder_path = 'image/'
images = glob.glob(folder_path + '*.png')
for image in images:
    img = np.float32(cv.imread(image, 0))
    showDFFT(img)

