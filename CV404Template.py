import numpy as np
from scipy.signal import correlate2d
from time import time
import matplotlib.pyplot as plt
from PIL import Image



##Idx is a lit of coordinates where idx[0] is 1st coordinate and idx[0][0] is row (y) and idx[0][1] is col (x)

def match_template_corr( x , template , numPeaks = None, thresh = None):
    plt.imshow(x)
    plt.imshow(template)
    start_time = time()
    y = np.empty(x.shape)
    y = correlate2d(x,template,'same')
    end_time = time()
    comp_time_ms = end_time-start_time
    idx = find_peaks(y, template, numPeaks, thresh) #row then column
    print(idx)
    return y, idx, comp_time_ms

def match_template_corr_zmean( x , template , numPeaks = None, thresh = None):
    y, idx, comp_time_ms = match_template_corr(x , template - template.mean(), numPeaks, thresh)
    return y, idx, comp_time_ms

def match_template_ssd( x , template , numPeaks = None, thresh = None):
    start_time = time()
    term1 = np.sum( np.square( template ))
    term2 = -2*correlate2d(x, template,'same')
    term3 = correlate2d( np.square( x ), np.ones(template.shape),'same' )
    ssd = np.maximum( term1 + term2 + term3 , 0 )
    y = 1 - np.sqrt(ssd)
    end_time = time()
    comp_time_ms = end_time-start_time
    idx = find_peaks(y, template, numPeaks, thresh) #row then column
    print(idx)
    return y, idx, comp_time_ms

def match_template_xcorr( f , template , numPeaks = None, thresh = None):
    start_time = time()
    t = template
    f_c = f - correlate2d( f , np.ones(t.shape)/np.prod(t.shape), 'same') 
    t_c = t - t.mean()
    numerator = correlate2d( f_c , t_c , 'same' )
    d1 = correlate2d( np.square(f_c) , np.ones(t.shape), 'same')
    d2 = np.sum( np.square( t_c ))
    denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
    response = np.zeros( f.shape )
    valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
    response[valid] = numerator[valid]/denumerator[valid]
    end_time = time()
    comp_time_ms = end_time-start_time
    idx = find_peaks(response, template, numPeaks, thresh) #row then column
    return response, idx, comp_time_ms

def find_peaks(matches, template, numPeaks, thresh):
    if numPeaks == None:
        numPeaks = 1
    if thresh == None:
        thresh = 0.5 * matches.max()
    nHoodSize = [template.shape[0],template.shape[1]] #height and width of template
    done = False
    row = []
    col = []
    newMatches = matches.copy()
    if thresh >= 0:
        cond = 0
    else:
        cond = 1
    while not done:
        matches_max = np.amax(newMatches) 
        max_indices = np.where(newMatches == matches_max)
        p = max_indices[0][0]
        q = max_indices[1][0]
        if cond == 0:
            if newMatches[p, q] >= thresh:
                row.append(p)
                col.append(q)
                p1 = int(p - (nHoodSize[0] - 1)/2 )
                p2 = int(p + (nHoodSize[0] - 1)/2)
                q1 = int(q - (nHoodSize[1] - 1)/2)
                q2 = int(q + (nHoodSize[1] - 1)/2)
                pp,qq = np.mgrid[p1:p2, q1:q2]
                pp = np.ravel(pp,'F')
                qq = np.ravel(qq,'F')
                gridCoordinates = list(zip(pp,qq))
                try:
                    for i in range(len(gridCoordinates)):
                        newMatches[gridCoordinates[i]] = 0.0
                except IndexError:
                    continue
                done = (len(row) == numPeaks)
            else:
                done = True
        elif cond == 1:
            if newMatches[p, q] <= thresh:
                row.append(p)
                col.append(q)
                p1 = int(p - (nHoodSize[0] - 1)/2 )
                p2 = int(p + (nHoodSize[0] - 1)/2)
                q1 = int(q - (nHoodSize[1] - 1)/2)
                q2 = int(q + (nHoodSize[1] - 1)/2)
                pp,qq = np.mgrid[p1:p2, q1:q2]
                pp = np.ravel(pp,'F')
                qq = np.ravel(qq,'F')
                gridCoordinates = list(zip(pp,qq))
                try:
                    for i in range(len(gridCoordinates)):
                        newMatches[gridCoordinates[i]] = 0.0
                except IndexError:
                    continue
                done = (len(row) == numPeaks)
            else:
                done = True    
    return list(zip(row, col))

##### For plotting in jupyter notebook
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(imgs_gray[1], cmap = 'gray')
# ax[1].imshow(matches_xcorr[1], cmap='gray')
# htemp, wtemp = templates[1].shape
# for f in range(len(idx)):
#     # x is c and r is y
#     r = idx[f][0]=>height
#     c = idx[f][1]=>width
#     if f == 0:
#         rect = plt.Rectangle((c-wtemp/2, r-htemp/2), wtemp, htemp, edgecolor='r', facecolor='none')
#         ax[1].plot(c, r, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#     else:
#         rect =  plt.Rectangle((c-wtemp/2, r-htemp/2), wtemp, htemp, edgecolor='g', facecolor='none')
#         ax[1].plot(c, r, 'o', markeredgecolor='g', markerfacecolor='none', markersize=10)
#     ax[0].add_patch(rect)

def rgb2gray(rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

