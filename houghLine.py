from collections import defaultdict
import numpy as np
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
    show_lines( img, accumulator , thetas , rhos, row, col )


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
    return lines,acc2



def show_lines( img, accumulator , thetas , rhos, row, col ):

    lines_img2, acc2_img2 = extract_lines( accumulator , thetas , rhos, row, col ) 
    
    fig = plt.figure(figsize=(20,20))    
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    
    
    limits = [rhos[0], rhos[-1], np.rad2deg(thetas[0]), np.rad2deg(thetas[-1])]
    ax1.set_title('Hough Space')
    ax1.imshow(accumulator, aspect='auto', extent= limits, cmap=cm.hot,interpolation='bilinear' )
    ax1.set_ylabel('Theta')
    ax1.set_xlabel('Rho')
    ax1.set_title('Hough')
    
    ax2.set_title('Hough Space (Processed)')
    ax2.imshow(acc2_img2, aspect='auto', extent= limits, cmap=cm.hot,interpolation='bilinear' )
    ax2.set_ylabel('Theta')
    ax2.set_xlabel('Rho')
    
    im = ax3.imshow( img, cmap='gray' )
    ax3.set_title('Original Image /w lines')
    ax3.autoscale(False)
    
    for (rho,theta), val in lines_img2.items():
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        ax3.plot((x0,x1), (y0, y1), '-r')

    plt.show()


hough_line_transform(np.array(Image.open('houghTestImages/Chess_Board.png')))
