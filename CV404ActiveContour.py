import os
import sys
import cv2 as cv2
import matplotlib.cm as cm
from scipy.ndimage import filters
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
import copy
from scipy import ndimage
import scipy.ndimage
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from CV404Filters import sobel_edge


#internal energy
def internalEnergy(snake, alpha, beta):
    iEnergy=0
    snakeLength=len(snake)
    for index in range(snakeLength-1,-1,-1):  #??
        nextPoint = (index+1)%snakeLength
        currentPoint = index % snakeLength
        previousePoint = (index - 1) % snakeLength
        iEnergy = iEnergy+ (alpha *(np.linalg.norm(snake[nextPoint] - snake[currentPoint] )**2))\
                  + (beta * (np.linalg.norm(snake[nextPoint] - 2 * snake[currentPoint] + snake[previousePoint])**2))
    return iEnergy

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)
#Gradient
def basicImageGradiant(image):
    #s_mask = 17
    #sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=s_mask))
    #sobelx = interval_mapping(sobelx, np.min(sobelx), np.max(sobelx), 0, 255)
    #sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=s_mask))
    #sobely = interval_mapping(sobely, np.min(sobely), np.max(sobely), 0, 255)



    gradient = cv2.Canny(image,100,400)

    #Gy, Gx = np.gradient(image)
    #Gx = interval_mapping(Gx, np.min(Gx), np.max(Gx), 0, 255)
    #Gy = interval_mapping(Gy, np.min(Gy), np.max(Gy), 0, 255)
    #gradient = 0.5 * Gx + 0.5 * Gy

    #gradient, direction = sobel_edge(image)

    #gradient = 0.5 * sobelx + 0.5 * sobely
    #print(sobelx)
    #print(sobely)
    return gradient

def imageGradient(gradient, snak):
    sum = 0
    snaxels_Len= len(snak)
    for index in range(snaxels_Len-1):
        point = snak[index]
        sum = sum+((gradient[point[1]][point[0]]))
    return sum

#External energy
def externalEnergy(grediant,image,snak):
    sum = 0
    snaxels_Len = len(snak)
    for index in range(snaxels_Len - 1):
        point = snak[index]
        sum = +(image[point[1]][point[0]])
    pixel = 255 * sum
    eEnergy = 40 * (pixel - imageGradient(grediant, snak)) 
    return eEnergy

#Total energy
def totalEnergy(grediant, image, snake, alpha, beta, gamma):
    iEnergy = internalEnergy(snake, alpha, beta)
    eEnergy=externalEnergy(grediant, image, snake)
    tEnergy = iEnergy+(gamma * eEnergy)
    return tEnergy

#Draw circle
def _pointsOnCircle(center, radius, num_points=12):
    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i)/num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p
        
    return points

def isPointInsideImage(image, point):

    return np.all(point < np.shape(image)) and np.all(point > 0)

#Apply contour
def activeContour(image_file, center, radius, alpha =300, beta =2, gamma =50, iter = 100):
    neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])
    image = cv2.imread(image_file, 0)
    print(image.shape)
    snake = _pointsOnCircle(center, radius, 30)
    grediant = basicImageGradiant(image)

    snakeColon =  copy.deepcopy(snake)

    for i in range(iter):
        for index,point in enumerate(snake):
            min_energy2 = float("inf")
            for cindex,movement in enumerate(neighbors):
                next_node = (point + movement)
                if not isPointInsideImage(image, next_node):
                    continue
                if not isPointInsideImage(image, point):
                    continue

                snakeColon[index]=next_node

                totalEnergyNext = totalEnergy(grediant, image, snakeColon, alpha, beta, gamma)

                if (totalEnergyNext < min_energy2):
                    min_energy2 = copy.deepcopy(totalEnergyNext)
                    indexOFlessEnergy = copy.deepcopy(cindex)
            snake[index] = (snake[index]+neighbors[indexOFlessEnergy])
        snakeColon = copy.deepcopy(snake)
        print(i)
    return image, snake

