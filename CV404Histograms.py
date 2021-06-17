from os import listdir
from os.path import isfile , join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from skimage import io
from skimage import color
from numpy import asarray


def img_redund(img):
    image_redund = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            image_redund[i][j] = reduced(img[i][j], 3)
    print(image_redund)
    return image_redund



def NormalizeImage(img):
    print(img)
    cum_arr = np.array(img.copy())
    nj = (cum_arr - cum_arr.min()) * 255
    N = cum_arr.max() - cum_arr.min()

# re-normalize the cdf
    cum_arr = nj / N
    cum_arr = cum_arr.astype('uint8')
    #cum_arr.append(0)
    return cum_arr



def reduced(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier



def get_hist(img):
    image_hist = img_redund(img)
    values = [np.max(image_hist),np.min(image_hist)]
    numbers = [1,1]
    for i in range(image_hist.shape[0]):
        for j in range(image_hist.shape[1]):
            if image_hist[i][j] in values:
                numbers[values.index(image_hist[i][j])] = numbers[values.index(image_hist[i][j])] + 1
            else:
                values.append(image_hist[i][j])
                numbers.append(1)
    return values, numbers

def get_hist_1(img, bins):
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in img:
        histogram[pixel] += 1
    
    # return our final result
    return histogram, np.arange(len(histogram))


def cumulativeSum(arr):
    arr1 = iter(arr)
    cumulative_sum = [next(arr1)]
    for i in arr1:
        cumulative_sum.append(cumulative_sum[-1]+i)

    return np.array(cumulative_sum)

def normalize_equalization(img):
    cum_arr = cumulativeSum(img)
    print(cum_arr.shape)
    nj = (cum_arr - cum_arr.min()) * 255
    N = cum_arr.max() - cum_arr.min()

# re-normalize the cdf
    cum_arr = nj / N
    cum_arr = cum_arr.astype('uint8')
    return cum_arr

def equalize_histogram_1(img, flat):
    equalized_img = normalize_equalization(img)
    
    return equalized_img[flat]



def get_probability(arr):
    prob_arr = np.zeros(len(arr))
    for i in range(len(arr)):
        prob_arr[i] = arr[i]/np.sum(arr)
    return prob_arr



def get_cumulative_prob(arr):
    probability_arr = get_probability(arr)
    cumulative_prob_arr = np.zeros(len(arr))
    for i in range(len(probability_arr)):
        sum = 0
        for j in range(i):
            sum = sum + probability_arr[j] 
        cumulative_prob_arr[i] = sum
    return cumulative_prob_arr



def equalize_histogram(values_freq, intensity = 255):
    new_values = get_cumulative_prob(values_freq)
    return np.floor(new_values * intensity)



def image_mapping(img, modified_data, equalized_data):
    image_mapped = img_redund(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(len(modified_data)):
                if image_mapped[i][j] == modified_data[k]:
                    image_mapped[i][j] = equalized_data[k]
                    break
    return image_mapped



def normalize_hist(ie):
    normalized_i = np.zeros(len(ie))
    min_val = np.min(ie)
    max_val = np.max(ie)
    for i in range(len(ie)):
        if ie[i] < min_val:
            min_val = ie[i]
        if ie[i] > max_val:
            max_val = ie[i]
    scale = 255.0/(max_val - min_val)
    for i in range(len(ie)):
        normalized_i[i] = scale*(ie[i] - min_val)
    return normalized_i


def get_threshold(img, thresh=20,stop=1.0):
    if(len(img.shape) == 3):
        x_low, y_low, z_meh = np.where(img<=thresh)
        x_high, y_high, z_mehh = np.where(img>thresh)    
        mean_low = np.mean(img[x_low,y_low])
        mean_high = np.mean(img[x_high,y_high])
        new_thresh = (mean_low + mean_high)/2
        if abs(new_thresh-thresh)< stop:
            return new_thresh
        else:
            return get_threshold(img, thresh=new_thresh,stop=1.0)
    else:
        x_low, y_low = np.where(img<=thresh)
        x_high, y_high = np.where(img>thresh)    
        mean_low = np.mean(img[x_low,y_low])
        mean_high = np.mean(img[x_high,y_high])
        new_thresh = (mean_low + mean_high)/2
        if abs(new_thresh-thresh)< stop:
            return new_thresh
        else:
            return get_threshold(img, thresh=new_thresh,stop=1.0)


def global_threshold(img):
    thresh_value = get_threshold(img, img.mean(), stop = 1.0)
    val_high = np.max(img)
    val_low = np.min(img)
    img_thresh = img.copy()
    if(len(img.shape) == 3):
        k = np.nonzero(img[0,0])[0][0]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,k] > thresh_value:
                    img_thresh[i,j,k] = val_high
                else:
                    img_thresh[i,j,k] = val_low
        return img_thresh
    if(len(img.shape) == 2):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > thresh_value:
                    img_thresh[i,j] = val_high
                else:
                    img_thresh[i,j] = val_low
        return img_thresh