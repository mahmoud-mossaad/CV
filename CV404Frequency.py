import matplotlib.pyplot as plt
from matplotlib import cm
#from skimage.color import rgb2gray
#from skimage.io import imread
import scipy.fftpack as fp
import numpy as np
from scipy import signal
from PIL import Image


#low pass filter

#Gaussian kernel
def low_pass_filter(im):
     im1 = rgb2gray(im)
     kernel = np.outer(signal.gaussian(im1.shape[0], 4), signal.gaussian(im1.shape[1], 4))
     freq = fp.fft2(im1)
     assert(freq.shape == kernel.shape) #check the condition
     freq_kernel = fp.fft2(fp.ifftshift(kernel)) 
     convolved = freq*freq_kernel
#plt.figure(figsize=(10,10))
#plt.imshow( (20*np.log10( 0.1 + freq)).astype(int), cmap=plt.get_cmap('gray'))

#Inverse fourier transform
     im_blur = fp.ifft2(convolved).real
     im_blur = 255 * im_blur / np.max(im_blur)
#plt.figure(figsize=(10,10))
#plt.imshow( (20*np.log10( 0.01 + fp.fftshift(freq_kernel))).astype(int), cmap=plt.get_cmap('gray'))
#plt.colorbar()
#plt.show()


#plt.figure(figsize=(10,10))
#plt.imshow(im_blur, cmap=plt.get_cmap('gray'))
#plt.show()
     return(im_blur)

#Fourier transform
def high_pass_filter(img):
     img1 = rgb2gray(img)
     F1 = fp.fft2((img1).astype(float))
     F2 = fp.fftshift(F1)
#plt.figure(figsize=(10,10))
#plt.imshow( (20*np.log10( 0.1 + F2)).astype(int), cmap=plt.get_cmap('gray'))
#plt.show()

#Block low frequency
     (w, h) = img1.shape
     half_w, half_h = int(w/2), int(h/2)
     n = 25
     F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
#print(img.shape)
#plt.figure(figsize=(10,10))
#plt.imshow( (20*np.log10( 0.1 + F2)).astype(int),cmap=plt.get_cmap('gray'))
#plt.colorbar()
#plt.show()

#Inverse fourier transform 
     im1 = fp.ifft2(fp.ifftshift(F2)).real
#plt.figure(figsize=(10,10))
#plt.imshow(im1, cmap='gray')
#plt.show()
     return(im1)

#hybrid image
def hybrid(img, im):
      print("High pass image", im.size)
      print("Low pass image", im.size)
      new_img1, new_img2 = resized_images(img,im)
      Hybrid= high_pass_filter(np.asarray(new_img1)) + low_pass_filter(np.asarray(new_img2))
#plt.figure(figsize=(10,10))
#print(Hybrid)
#plt.imshow(Hybrid,cmap=plt.get_cmap('gray'))
      return(Hybrid)


def resized_images(image1, image2):
        sizes_x = np.array([image1.size[0], image2.size[0]])
        sizes_y = np.array([image1.size[1], image2.size[1]])
        new_x = sizes_x.min()
        new_y = sizes_y.min()
        im_hybrid_1 = image1.resize((new_x, new_y), Image.ANTIALIAS)
        im_hybrid_2 = image2.resize((new_x, new_y), Image.ANTIALIAS)

        return im_hybrid_1, im_hybrid_2



def rgb2gray(rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])