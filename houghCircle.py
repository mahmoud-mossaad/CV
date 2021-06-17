from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

def gray_scale(img):
    r = img[:,:,0] * 0.2989 
    g = img[:,:,1] * 0.5870 
    b = img[:,:,2] * 0.1140
    return np.add(r, g, b)


def hough_transform_circle(img, rmin = 18, threshold = 0.35, steps = 100):
    imGS = gray_scale( img )
    imGB = cv2.GaussianBlur(imGS,(11,11),1)
    edge = cv2.Canny(imGB.astype(np.uint8),100,200)
    rmax = np.max((img.shape[0],img.shape[1]))
    circles = hough_circle(edge, rmin, rmax, threshold, steps)
    return circles

def hough_circle(img, rmin, rmax, threshold, steps):

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * np.cos(2 * np.pi * t / steps)), int(r * np.sin(2 * np.pi * t / steps))))

    print(len(points))
    print(img.shape)
    acc = defaultdict(int)
    for x, y in np.argwhere(img[:,:]):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v/steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))
    return circles
