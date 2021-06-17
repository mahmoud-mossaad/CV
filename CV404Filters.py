import numpy as np
import scipy as sci
from scipy import misc
from scipy import signal

def map_img(img):
    imgRange = img.max() - img.min()
    imgMapped = np.subtract(img, img.min())*255/imgRange
    return imgMapped

def normalize_img(mode, img):
    if (mode == "01"):
        normalizedImg = np.subtract(img, img.min())/(img.max()-img.min())
    elif (mode == "std"):
        normalizedImg = np.subtract(img, img.mean())/(img.std())
    #elif (mode == "allImg"):
     #   normalizedImg = np.subtract(img, img.mean())/(img.std())
    return normalizedImg    

################# NOISE ADDITION #######################################################################################

def uniform_noise(img):
    noise = np.random.uniform(low = -128, high = 128, size = img.shape)
#     noise = map_img(noise)
    imgNoisy = np.add(img, noise)
    imgNoisyMapped = map_img(imgNoisy)
    return imgNoisyMapped

def gaussian_noise(img, mu = 0.0, std = 1.0):
    size = img.shape
    noise = np.random.normal(loc = mu, scale = std, size = img.shape)
    noise = map_img(noise)
    imgNoisy = np.add(img, noise)
    imgNoisyMapped = map_img(imgNoisy)
    return imgNoisyMapped

def salt_n_pepper(img, saltPercent, origPercent):
    pepperPercent = 100-saltPercent-origPercent
    percentilesIndx = np.random.rand(img.shape[0], img.shape[1]) #values between 0 and 1
    saltIndx = np.where(percentilesIndx < saltPercent/100)
    pepperIndx = np.where(percentilesIndx < pepperPercent/100)
    imgNoisy = img
    imgNoisy[saltIndx] = 255
    imgNoisy[pepperIndx] = 0
    return imgNoisy

################# NOISE FILTERING ###################################################################################

def average_filter(img, kernelSize = 3):
    fltr = np.ones((kernelSize,kernelSize), dtype=int) * 1/(kernelSize * kernelSize)
    cleanImg = signal.convolve2d(img, fltr, mode = "same") #o/p same size as img, default padding zero
    return cleanImg

def gaussian_filter(img, sigma = 1.0, kernelSize = 3):
    M = kernelSize*kernelSize
    fltrWindow = np.asmatrix(signal.gaussian(M, std = sigma, sym=True))
    fltrT = np.transpose(fltrWindow)
    kernel = np.matmul(fltrT,fltrWindow)
    kernelNormalized = kernel/np.sum(kernel)
    cleanImg = signal.convolve2d(img, kernelNormalized, mode = "same") #o/p same size as img, default padding zero
    return cleanImg

def median_filter(img, kernelSize = 3): 
    fltr = [0]*kernelSize*kernelSize
    cleanImg = np.zeros(img.shape, dtype = int)
    imgRow = img.shape[0]
    imgCol = img.shape[1]
    boundary = (kernelSize // 2)
    for x in range (boundary, imgRow-boundary):
        for y in range (boundary, imgCol-boundary):
            i = 0
            for fltrx in range(0, kernelSize): 
                for fltry in range (0, kernelSize):
                    fltr[i] = img[x + fltrx - boundary][y + fltry - boundary]
                    i = i + 1
            fltr.sort()
            cleanImg[x][y] = fltr[kernelSize * kernelSize // 2]
    return cleanImg

################# EDGE DETECTION ####################################################################################

def sobel_edge(img, threshold = 0, mode = 'gray'):
    kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    grady = signal.convolve2d(img, kernely, mode = "same") #o/p same size as img, default padding zero
    gradx = signal.convolve2d(img, kernelx, mode = "same")
#     edgeMagnitude = abs(gradx) + abs(grady)
    edgeMagnitude = np.sqrt(np.square(gradx) + np.square(grady))
#     edgeMagnitude = np.hypot(gradx, grady)
    edgeDirection = np.arctan2(grady, gradx)
#     print(edgeMagnitude.max())
#     print(edgeMagnitude.min())
    edgeMagnitude *= 255.0/ edgeMagnitude.max()
#     edgeMagnitude = map_img(edgeMagnitude) #scale mag from 0 to 255
#     edgeMagnitude = np.round(edgeMagnitude)
#     threshold = 70 #[0 255]
    if(mode == 'binary'and threshold != 0):
        edgeMagnitude[edgeMagnitude > threshold] = 255
        edgeMagnitude[edgeMagnitude <= threshold] = threshold #or zero
    return edgeMagnitude, edgeDirection

def roberts_edge(img, threshold = 0, mode = 'gray'):
    kernely = np.array([[1, 0], [0, -1]], np.float32)
    kernelx = np.array([[0, 1], [-1, 0]], np.float32)
    grady = signal.convolve2d(img, kernely, mode = "same") #o/p same size as img, default padding zero
    gradx = signal.convolve2d(img, kernelx, mode = "same")
#     edgeMagnitude = abs(gradx) + abs(grady)
    edgeMagnitude = np.sqrt(np.square(gradx) + np.square(grady))
    edgeDirection = np.arctan2(grady, gradx)
#     print(edgeMagnitude.max())
#     print(edgeMagnitude.min())
#     edgeMagnitude *= 255/ edgeMagnitude.max()
    edgeMagnitude = map_img(edgeMagnitude)
#     edgeMagnitude = np.round(edgeMagnitude)
#     threshold = 70 #[0 255]
    if(mode == 'binary'and threshold != 0):
        edgeMagnitude[edgeMagnitude > threshold] = 255
        edgeMagnitude[edgeMagnitude <= threshold] = 0
    return edgeMagnitude, edgeDirection

def prewitt_edge(img, threshold = 0, mode = 'gray'):
    kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32)
    kernelx = np.transpose(kernely)
    grady = signal.convolve2d(img, kernely, mode = "same") #o/p same size as img, default padding zero
    gradx = signal.convolve2d(img, kernelx, mode = "same")
#     edgeMagnitude = abs(gradx) + abs(grady)
    edgeMagnitude = np.sqrt(np.square(gradx) + np.square(grady))
    edgeDirection = np.arctan2(grady, gradx)
#     print(edgeMagnitude.max())
#     print(edgeMagnitude.min())
#     edgeMagnitude *= 255/ edgeMagnitude.max()
    edgeMagnitude = map_img(edgeMagnitude) #scale mag from 0 to 255
#     edgeMagnitude = np.round(edgeMagnitude)
#     threshold = 70 #[0 255]
    if(mode == 'binary'and threshold != 0):
        edgeMagnitude[edgeMagnitude <= threshold] = 0
        edgeMagnitude[edgeMagnitude > threshold] = 255
    return edgeMagnitude, edgeDirection

################################################################################################################

def non_max_suppression(magnitude, direction):
    row, col = magnitude.shape
    suppressed = np.zeros((row,col))
    angle = np.rad2deg(direction)
    angle += 180

    for i in range(1,row-1):
        for j in range(1,col-1):
            try:
                beforePixel = 255
                afterPixel = 255
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (337.5 <= angle[i,j] <= 360):
                    afterPixel = magnitude[i, j+1]
                    beforePixel = magnitude[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5) or (202.5 <= angle[i][j] < 247.5):
                    afterPixel = magnitude[i+1, j-1]
                    beforePixel = magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i][j] < 112.5) or (247.5 <= angle[i][j] < 292.5):
                    afterPixel = magnitude[i+1, j]
                    beforePixel = magnitude[i-1, j]
                #angle 135
                else:
                    afterPixel = magnitude[i-1, j-1]
                    beforePixel = magnitude[i+1, j+1]

                if (magnitude[i,j] >= beforePixel) and (magnitude[i,j] >= afterPixel):
                    suppressed[i,j] = magnitude[i,j]

            except IndexError as e:
                pass
    
    return suppressed

def double_threshold(img, low, high, weak):
    
    row, col = img.shape
    thresholded = np.zeros((row,col))
    
    strong = 255
    
    strong_i, strong_j = np.where(img >= high)
#     zeros_i, zeros_j = np.where(img < low)
    
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    
    thresholded[strong_i, strong_j] = strong
    thresholded[weak_i, weak_j] = weak
    
    return (thresholded, weak, strong)

def hysteresis(img, weak, strong=255):
    row, col = img.shape
    image_row, image_col= img.shape
    top = img.copy()
    for i in range(1, row-1):
        for j in range(1, col-1):
            if (top[i,j] == weak):
                try:
                    if ((top[i+1, j-1] == strong) or (top[i+1, j] == strong) or (top[i+1, j+1] == strong)
                        or (top[i, j-1] == strong) or (top[i, j+1] == strong)
                        or (top[i-1, j-1] == strong) or (top[i-1, j] == strong) or (top[i-1, j+1] == strong)):
                        top[i, j] = strong
                    else:
                        top[i, j] = 0
                except IndexError as e:
                    pass
    final_image = top
    return final_image

def canny_edge(img, sigma = 0.1, gaussSize = 3, filter = sobel_edge, minThresh = 5, maxThresh = 20):
    
    weak = 50
    smoothed_img = gaussian_filter(img, sigma, gaussSize)
    grad, theta = filter(smoothed_img)
    suppressed = non_max_suppression(grad, theta)
    thresholded, weak, strong = double_threshold(suppressed, minThresh, maxThresh, weak)
    hyster = hysteresis(thresholded, weak)

    return hyster
