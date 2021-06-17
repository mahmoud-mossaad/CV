from scipy.signal import convolve2d
from cvutils import gaussian_kernel2d
from skimage.transform import rescale
from math import sqrt
import sys
import numpy as np
from PIL import Image
from cvutils import rgb2gray,sift_resize
from cvutils import padded_slice, sift_gradient
import matplotlib.cm as cm
import cv2
from skimage.transform import resize
import itertools
import time
from math import sin, cos
from skimage.transform import rotate


class SIFT:
    def __init__(self, n_octaves = 4, n_scales = 5, sigma = 1.6, k = sqrt(2)):
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma = sigma
        self.k = k
        self.SIGMA_SEQ = lambda s: [ (self.k**i)*s for i in range(self.n_scales) ]
        self.SIGMA_SIFT = self.SIGMA_SEQ(self.sigma) #
        self.KERNEL_RADIUS = lambda s : 2 * int(round(s))
        self.KERNELS_SIFT = [ gaussian_kernel2d(std = s, kernlen = 2 * self.KERNEL_RADIUS(s) + 1) for s in self.SIGMA_SIFT ]
        self.filename1 = None
        self.filename2 = None


    def read_image(self, image1, image2):
        self.filename1 = image1
        self.filename2 = image2

    def image_dog(self, img ):
        octaves = []
        dog = []
        base = rescale( img, 2, anti_aliasing=False) 
        octaves.append([ convolve2d( base , kernel , 'same', 'symm') 
                        for kernel in self.KERNELS_SIFT ])
        dog.append([ s2 - s1 
                    for (s1,s2) in zip( octaves[0][:-1], octaves[0][1:])])
        for i in range(1,self.n_octaves):
            base = octaves[i-1][2][::2,::2] # 2x subsampling 
            octaves.append([base] + [convolve2d( base , kernel , 'same', 'symm') 
                                    for kernel in self.KERNELS_SIFT[1:] ])
            dog.append([ s2 - s1 
                        for (s1,s2) in zip( octaves[i][:-1], octaves[i][1:])])
        return dog , octaves

    
    def rotated_subimage(self,image, center, theta, width, height):
        theta *= 3.14159 / 180 # convert to rad
        
        
        v_x = (cos(theta), sin(theta))
        v_y = (-sin(theta), cos(theta))
        s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
        s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

        mapping = np.array([[v_x[0],v_y[0], s_x],
                            [v_x[1],v_y[1], s_y]])

        return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)



    def cube_extrema(self, img1, img2, img3 ):
        value = img2[1,1]

        if value > 0:
            return all([np.all( value >= img ) for img in [img1,img2,img3]]) # test map
        else:
            return all([np.all( value <= img ) for img in [img1,img2,img3]]) # test map



    def contrast(self, dog , img_max, threshold = 0.03 ):
        dog_norm = dog / img_max
        coords = list(map( tuple , np.argwhere( np.abs( dog_norm ) > threshold ).tolist() ))
        return coords


    
    def corners(self, dog , r = 10 ):
        threshold = ((r + 1.0)**2)/r
        dx = np.array([-1,1]).reshape((1,2))
        dy = dx.T
        dog_x = convolve2d( dog , dx , boundary='symm', mode='same' )
        dog_y = convolve2d( dog , dy , boundary='symm', mode='same' )
        dog_xx = convolve2d( dog_x , dx , boundary='symm', mode='same' )
        dog_yy = convolve2d( dog_y , dy , boundary='symm', mode='same' )
        dog_xy = convolve2d( dog_x , dy , boundary='symm', mode='same' )
        
        tr = dog_xx + dog_yy
        det = dog_xx * dog_yy - dog_xy ** 2
        response = ( tr**2 +10e-8) / (det+10e-8)
        
        coords = list(map( tuple , np.argwhere( response < threshold ).tolist() ))
        return coords


    def dog_keypoints(self, img_dogs , img_max , threshold = 0.03 ):
        octaves_keypoints = []
        
        for octave_idx in range(self.n_octaves):
            img_octave_dogs = img_dogs[octave_idx]
            keypoints_per_octave = []
            for dog_idx in range(1, len(img_octave_dogs)-1):
                dog = img_octave_dogs[dog_idx]
                keypoints = np.full( dog.shape, False, dtype = np.bool)
                candidates = set( (i,j) for i in range(1, dog.shape[0] - 1) for j in range(1, dog.shape[1] - 1))
                search_size = len(candidates)
                candidates = candidates & set(self.corners(dog)) & set(self.contrast( dog , img_max, threshold ))
                search_size_filtered = len(candidates)
                for i,j in candidates:
                    slice1 = img_octave_dogs[dog_idx -1][i-1:i+2, j-1:j+2]
                    slice2 = img_octave_dogs[dog_idx   ][i-1:i+2, j-1:j+2]
                    slice3 = img_octave_dogs[dog_idx +1][i-1:i+2, j-1:j+2]
                    if self.cube_extrema( slice1, slice2, slice3 ):
                        keypoints[i,j] = True
                keypoints_per_octave.append(keypoints)
            octaves_keypoints.append(keypoints_per_octave)
        return octaves_keypoints



    def dog_keypoints_orientations(self, img_gaussians , keypoints , num_bins = 36 ):
        kps = []
        for octave_idx in range(self.n_octaves):
            img_octave_gaussians = img_gaussians[octave_idx]
            octave_keypoints = keypoints[octave_idx]
            for idx,scale_keypoints in enumerate(octave_keypoints):
                scale_idx = idx + 1 ## idx+1 to be replaced by quadratic localization
                gaussian_img = img_octave_gaussians[ scale_idx ] 
                sigma = 1.5 * self.sigma * ( 2 ** octave_idx ) * ( self.k ** (scale_idx))
                radius = self.KERNEL_RADIUS(sigma)
                kernel = gaussian_kernel2d(std = sigma, kernlen = 2 * radius + 1)
                gx,gy,magnitude,direction = sift_gradient(gaussian_img)
                direction_idx = np.round( direction * num_bins / 360 ).astype(int)          
                
                for i,j in map( tuple , np.argwhere( scale_keypoints ).tolist() ):
                    window = [i-radius, i+radius+1, j-radius, j+radius+1]
                    mag_win = padded_slice( magnitude , window )
                    dir_idx = padded_slice( direction_idx, window )
                    weight = mag_win * kernel 
                    hist = np.zeros(num_bins, dtype=np.float32)
                    
                    for bin_idx in range(num_bins):
                        hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
                
                    for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():
                        angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
                        kps.append( (i,j,octave_idx,scale_idx,angle))
        return kps



    def extract_sift_descriptors128(self, img_gaussians, keypoints, num_bins = 8 ):
        descriptors = []; points = [];  data = {} # 
        for (i,j,oct_idx,scale_idx, orientation) in keypoints:

            if 'index' not in data or data['index'] != (oct_idx,scale_idx):
                data['index'] = (oct_idx,scale_idx)
                gaussian_img = img_gaussians[oct_idx][ scale_idx ] 
                sigma = 1.5 * self.sigma * ( 2 ** oct_idx ) * ( self.k ** (scale_idx))
                data['kernel'] = gaussian_kernel2d(std = sigma, kernlen = 16)                

                gx,gy,magnitude,direction = sift_gradient(gaussian_img)
                data['magnitude'] = magnitude
                data['direction'] = direction

            window_mag = self.rotated_subimage(data['magnitude'],(j,i), orientation, 16,16)
            window_mag = window_mag * data['kernel']
            window_dir = self.rotated_subimage(data['direction'],(j,i), orientation, 16,16)
            window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)

            features = []
            for sub_i in range(4):
                for sub_j in range(4):
                    sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                    sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                    hist = np.zeros(num_bins, dtype=np.float32)
                    for bin_idx in range(num_bins):
                        hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                    features.extend( hist.tolist())
            features = np.array(features) 
            features /= (np.linalg.norm(features))
            np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )
            assert features.shape[0] == 128, "features missing!"
            features /= (np.linalg.norm(features))
            descriptors.append(features)
            points.append( (i ,j , oct_idx, scale_idx, orientation))
        return points , descriptors



    def pipeline(self, input_img ):
        img_max = input_img.max()
        dogs, octaves = self.image_dog( input_img )
        keypoints = self.dog_keypoints( dogs , img_max , 0.03 )
        keypoints_ijso = self.dog_keypoints_orientations( octaves , keypoints , 36 )
        points,descriptors = self.extract_sift_descriptors128(octaves , keypoints_ijso , 8)
        return points, descriptors


    def apply_sift(self):
        image_patterns = {}
        image_patterns_gray = {}
        images = {}
        images_gray = {}

        img = np.array(Image.open(self.filename1))
        img,ratio = sift_resize(img)
        images[self.filename1] = img
        image_patterns[self.filename1] = []

        pattern,_ = sift_resize(np.array(Image.open(self.filename2)), ratio )
        image_patterns[self.filename1].append(pattern)
        image_patterns[self.filename1].append(rotate(pattern, 90))

        images_gray[self.filename1] = rgb2gray( images[self.filename1] )
        image_patterns_gray[self.filename1] = [ rgb2gray( img ) for img in image_patterns[self.filename1] ]

        image_patterns_sift = {}
        images_sift = {}

        images_sift[self.filename1] = self.pipeline(images_gray[self.filename1])
        image_patterns_sift[self.filename1] = []
        for pattern in image_patterns_gray[self.filename1]:
            image_patterns_sift[self.filename1].append( self.pipeline( pattern ))

        img = images[self.filename1]
        img_sift = images_sift[self.filename1]
        for i in range(len(image_patterns[self.filename1])):
            pattern = image_patterns[self.filename1][i]
            pattern_sift = image_patterns_sift[self.filename1][i]
            matched_img = self.match(img, img_sift[0], img_sift[1], pattern, pattern_sift[0], pattern_sift[1])
        #self.match_image = matched_img
        return matched_img



    def kp_list_2_opencv_kp_list(self, kp_list):

        opencv_kp_list = []
        for kp in kp_list:
            opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)),
                                    y=kp[0] * (2**(kp[2]-1)),
                                    _size=kp[3],
                                    _angle=kp[4],
    #                                  _response=kp[IDX_RESPONSE],
    #                                  _octave=np.int32(kp[2]),
                                    # _class_id=np.int32(kp[IDX_CLASSID])
                                    )
            opencv_kp_list += [opencv_kp]

        return opencv_kp_list


    
    def match(self, img_a, pts_a, desc_a, img_b, pts_b, desc_b):
        img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
        
        desc_a = np.array( desc_a , dtype = np.float32 )
        desc_b = np.array( desc_b , dtype = np.float32 )

        pts_a = self.kp_list_2_opencv_kp_list(pts_a)
        pts_b = self.kp_list_2_opencv_kp_list(pts_b)

        # create BFMatcher object
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_a,desc_b,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.25*n.distance:
                good.append(m)

        img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)

        cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #plt.figure(figsize=(20,20))
        #plt.imshow(img_match)
        #plt.show()

        return img_match