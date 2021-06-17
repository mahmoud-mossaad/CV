from collections import defaultdict
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import math
from PIL import Image

def gray_scale(img):
    r = img[:,:,0] * 0.2989 
    g = img[:,:,1] * 0.5870 
    b = img[:,:,2] * 0.1140
    return np.add(r, g, b)

def hough_line_transform(img, thetaRes = 1, rhoRes = 1, numPeaks = 50,thresh = None, nHoodSize = None ):
    imGS = gray_scale( img )
    imGB = cv2.GaussianBlur(imGS,(11,11),1)
    edge = cv2.Canny(imGB.astype(np.uint8),100,200)
    accumulator, thetas, rhos = hough_space(edge, thetaRes, rhoRes)
    row, col, newHough = hough_peaks(accumulator, numPeaks, thresh, nHoodSize)
    points = show_lines( img, accumulator , thetas , rhos, row, col )
    return points


def hough_space(image, thetaRes = 1, rhoRes = 1): 
    
    #RhoRes in degrees
    
    Ny = image.shape[0] # rows
    Nx = image.shape[1] # columns
       
    maxRho = math.hypot(Nx, Ny)
    
    #Check resolution values
    if not(0 < thetaRes < 90):
        print('error please input a correct step for theta')
        return
        
    elif not(0 < rhoRes < maxRho):
        print('error please input a correct step for rho')
        return
    else:

        thetas = np.deg2rad(np.arange(-90.0, 90, thetaRes, dtype = float))
        thetasLen = len(thetas)
        
        ## Range of radius
        rhos = np.arange(-maxRho, maxRho, rhoRes, dtype = float)
        rhosLen = len(rhos)
        
        #2. Create accumulator array and initialize to zero
        accumulator = np.zeros((rhosLen, thetasLen))

        for y in range(Ny):
            for x in range(Nx):
                if image[y,x] > 0:
                    for k in range(thetasLen):
                        try:
                            r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                            ir = r/rhoRes 
                            if r >= 0:
                                accumulator[int(ir - rhosLen//2-1),k] += 1
                            else:
                                accumulator[int(ir + rhosLen//2-1),k] += 1
                        except IndexError:
                            pass
        return accumulator, thetas, rhos


def hough_peaks(accumulator, numPeaks, thresh = None, nHoodSize = None):
    if numPeaks == None:
        numPeaks = 1
    if thresh == None:
        thresh = 0.5 * accumulator.max()
    if nHoodSize == None:
        y = int (np.ceil(accumulator.shape[0]/50))
        x = int (np.ceil(accumulator.shape[1]/50))
        if y % 2 != 0:
            y = y + 1
        if x % 2 != 0:
            x = x + 1
        nHoodSize = [y,x]
    done = False
    row = []
    col = []
    newHough = accumulator.copy()
    while not done:
        hough_max = np.amax(newHough) 
        max_indices = np.where(newHough == hough_max)
        p = max_indices[0][0]
        q = max_indices[1][0]
        if newHough[p, q] >= thresh:
            row.append(p)
            col.append(q)
            p1 = int(p - (nHoodSize[0] - 1)/2 )
            p2 = int( p + (nHoodSize[0] - 1)/2)
            q1 = int(q - (nHoodSize[1] - 1)/2)
            q2 = int(q + (nHoodSize[1] - 1)/2)
            pp,qq = np.mgrid[p1:p2, q1:q2]
            pp = np.ravel(pp,'F')
            qq = np.ravel(qq,'F')
            gridCoordinates = list(zip(pp,qq))
            for i in range(len(gridCoordinates)):
                newHough[gridCoordinates[i]] = 0.0
            done = (len(row) == numPeaks)
        else:
            done = True
    return row, col, newHough



def extract_lines( accumulator , thetas , rhos, row, col) :
    lines = defaultdict()
    acc2 = np.zeros(accumulator.shape)
    for i in range(len(row)):
        for j in range(len(col)):
            theta = thetas[col[j]]
            rho = rhos[row[i]]
            lines[(rho,theta)] = accumulator[row[i], col[j]]
            acc2[row[i], col[j]] = accumulator[row[i], col[j]]
    return lines



def show_lines( img, accumulator , thetas , rhos, row, col ):

    lines = extract_lines( accumulator , thetas , rhos, row, col ) 

    points = []
    for (rho,theta), val in lines.items():
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        points.append([x0,x1,y0,y1])
    return points    
        # ax3.plot((x0,x1), (y0, y1), '-r')

def hough_transform_circle(img, rmin = 18, threshold = 0.35, steps = 100):
    imGS = gray_scale( img )
    imGB = cv2.GaussianBlur(imGS,(11,11),1)
    edge = cv2.Canny(imGB.astype(np.uint8),100,200)
    #(M,N) = img.shape
    rmax = np.max((img.shape[0],img.shape[1]))
    circles = hough_circle(edge, rmin, rmax, threshold, steps)
    return circles

def hough_circle(img, rmin, rmax, threshold, steps):

    points = []
    for r in range(int(rmin), int(rmax) + 1):
        for t in range(int(steps)):
            points.append((r, int(r * np.cos(2 * np.pi * t / steps)), int(r * np.sin(2 * np.pi * t / steps))))

    acc = defaultdict(int)
    for x, y in np.argwhere(img[:,:]):
        for r, dx, dy in points:
            if (x - dx, y - dy, r) in acc:
                acc[(x - dx, y - dy, r)] += 1
            else:
                acc[(x - dx, y - dy, r)] = 0

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v/steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))
    return circles