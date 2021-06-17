import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from CV404Filters import non_max_suppression
from scipy.interpolate import interp1d


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def threshold(img):

        highThreshold = img.max() * 0.15
        lowThreshold = highThreshold * 0.05

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(75)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)


def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def gray_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def get_magnitude_direction(im):
    kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    grady = signal.convolve2d(im, kernely, mode = "same")
    gradx = signal.convolve2d(im, kernelx, mode = "same")
    Magnitude = np.sqrt(np.square(gradx) + np.square(grady))
    Magnitude *= 255.0/ Magnitude.max()
    Direction = np.arctan2(grady, gradx)

    return Magnitude, Direction


def get_harris_corner(image, window = 5, k = 0.04, method = 'Thresholding' , param = 1000):
    image_gray = rgb2gray(image)
    print(image_gray.shape)
    if len(image.shape) > 2:
        color_img = gray_rgb(image_gray)
    else:
        color_img = gray_rgb(image)

    if method == "Non Maxima Supression":
        if len(image.shape) > 2:
            test_img = gray_rgb(image_gray)
        else:
            test_img = gray_rgb(image)
    
    if method == "Local Thresholding":
        if len(image.shape) > 2:
            test_img = gray_rgb(image_gray)
        else:
            test_img = gray_rgb(image)


    r_values = []
    ## Getting the height and width to iterate over the image
    height = int(image.shape[0])
    width = int(image.shape[1])
    #k = 0.04 ## Given
    #thresh = 1000 ## Given
    #window_size = 8 ## Given
    offset = int(window/2)


    ## Calculating the gradient
    if len(image.shape) > 2: 
        Gy, Gx = np.gradient(image_gray)
    else:
        Gy, Gx = np.gradient(image)

    Ixx = Gx**2
    Ixy = Gy*Gx
    Iyy = Gy**2

    for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                #Calculate sum of squares
                windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
                windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
                sum_Ixx = windowIxx.sum()
                sum_Ixy = windowIxy.sum()
                sum_Iyy = windowIyy.sum()

                #Getting the determent and trace to get the R
                det = (sum_Ixx * sum_Iyy) - (sum_Ixy**2)
                trace = sum_Ixx + sum_Iyy
                r = det - k*(trace**2)
                #print(r)
                #If corner response is over threshold, color the point and add to corner list
                if method == 'Thresholding':
                    if r > param*10000:
                        print(y,x)
                        color_img.itemset((y, x, 0), 255)
                        color_img.itemset((y, x, 1), 0)
                        color_img.itemset((y, x, 2), 0)
                if method == "Non Maxima Supression":
                    if r > 1000:
                        print(y,x)
                        r_values.append([y,x,r])
                if method == "Local Thresholding":
                    if r > 100:
                        print(y,x)
                        r_values.append([y,x,r])

    if method == "Local Thresholding":
        m = interp1d([0,max([r[-1] for r in r_values])],[0,255])
        for r in r_values:
            test_img.itemset((r[0], r[1], 0), float(m(r[2])))
        response = threshold(test_img[:,:,0])
        for y in range(test_img[:,:,0].shape[0]):
            for x in range(test_img[:,:,0].shape[1]):
                if response[y,x] < 255:
                    color_img.itemset((y,x,0), 255)
                    color_img.itemset((y,x,1), 0)
                    color_img.itemset((y,x,2), 0)
        return color_img

    if method == "Non Maxima Supression":
        m = interp1d([0,max([r[-1] for r in r_values])],[0,255])
        for r in r_values:
            test_img.itemset((r[0], r[1], 0), float(m(r[2])))
        m, d = get_magnitude_direction(test_img[:,:,0])
        final_img = non_max_suppression(m,d)
        
        
        for y in range(final_img.shape[0]):
            for x in range(final_img.shape[1]):
                if final_img[y,x] != 0:
                    color_img.itemset((y,x,0), final_img[y,x])
                    color_img.itemset((y,x,1), 0)
                    color_img.itemset((y,x,2), 0)
        return color_img

    print('done')
    return color_img