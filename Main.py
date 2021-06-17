import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtQuick, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import pyqtgraph as pg
from MainWindow import Ui_MainWindow
import qimage2ndarray
import cv2 as cv2
import matplotlib.image as mpimg
from PIL import Image
from skimage.io import imread
from skimage import io
from skimage import color
import scipy.fftpack as fp
from matplotlib import cm
import numpy as np
from skimage.color import rgb2hsv
from numpy import asarray
from scipy import signal
import math
from time import time
import matplotlib.pyplot as plt
from CV404Frequency import low_pass_filter, high_pass_filter, hybrid
from CV404Histograms import img_redund, reduced, get_hist, get_probability, get_cumulative_prob, equalize_histogram, image_mapping, normalize_hist, global_threshold, get_threshold, NormalizeImage, get_hist_1
from CV404Histograms import equalize_histogram_1, normalize_equalization, cumulativeSum
from CV404Filters import normalize_img, map_img
from CV404Filters import uniform_noise, gaussian_noise, salt_n_pepper, average_filter, gaussian_filter, median_filter, sobel_edge, roberts_edge, prewitt_edge, canny_edge
from CV404Harris import get_harris_corner, gray_rgb
from CV404ActiveContour import activeContour
from CV404Hough import hough_line_transform, hough_transform_circle
from CV404Template import match_template_corr, match_template_corr_zmean, match_template_ssd, match_template_xcorr
from CV404SIFT import SIFT


class window(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fileName_hybrid_1 = None
        self.fileName_hybrid_2 = None
        self.x = None
        self.y = None
        self.fileName_snake = None
        self.flag_draw = True
        self.counter = 0
        self.ui.filter_browse_button.clicked.connect(self.open_image_filter)
        self.ui.hist_browse_image.clicked.connect(self.open_image_hist)
        self.ui.hybrid_browse_image_1.clicked.connect(self.open_image_hybrid_1)
        self.ui.hybrid_browse_image_2.clicked.connect(self.open_image_hybrid_2)
        self.ui.hybrid_make_hybrid.clicked.connect(self.make_hybrid)
        self.ui.hist_combbox.currentIndexChanged.connect(self.draw_output_hist)
        self.ui.selectrgb.currentTextChanged.connect(self.draw_image_hist)
        self.ui.noise_combo_box.currentIndexChanged.connect(self.apply_noise)
        self.ui.filter_combo_box.currentIndexChanged.connect(self.apply_filter)
        self.ui.harris_load.clicked.connect(self.open_image_harriss)
        self.ui.harris_apply.clicked.connect(self.apply_harris)
        self.ui.harris_mode_select.currentIndexChanged.connect(self.apply_harris_variables)
        self.ui.snake_load_image.clicked.connect(self.open_snake_image)
        self.ui.snake_input_image.mousePressEvent = self.draw_circle_on_image
        self.ui.snake_clear_anchors.clicked.connect(self.clear_anchors)
        self.ui.snake_apply.clicked.connect(self.apply_contour)
        self.ui.hough_load.clicked.connect(self.open_image_hough)
        self.ui.hough_apply.clicked.connect(self.apply_hough)
        self.ui.lines_check_box.stateChanged.connect(self.change_hough_state)
        self.ui.circles_check_box.stateChanged.connect(self.change_hough_state_1)
        self.ui.Template_load_image_A.clicked.connect(self.open_image_template_A)
        self.ui.Template_load_image_B.clicked.connect(self.open_image_template_B)
        self.ui.sift_load_image_A.clicked.connect(self.open_image_sift_A)
        self.ui.sift_load_image_B.clicked.connect(self.open_image_sift_B)
        self.ui.sift_match_button.clicked.connect(self.apply_sift)
        self.ui.Template_match_button.clicked.connect(self.apply_detected_patterns)

    def change_hough_state(self):
        if self.ui.lines_check_box.isChecked():
            self.ui.circles_check_box.setChecked(False)

    def change_hough_state_1(self):
        if self.ui.circles_check_box.isChecked():
            self.ui.lines_check_box.setChecked(False)

    def apply_hough(self):
        if self.ui.lines_check_box.isChecked():
            points = hough_line_transform(self.image_hough, float(self.ui.hough_theta.text()), float(self.ui.hough_rho.text()),
            float(self.ui.hough_numpeaks.text()))
            points_1 = []
            for p in points:
                points_1.append(np.absolute(p))
            print(points_1)
            for p in points_1:
                image = cv2.line(self.image_hough, (int(p[0]), int(p[2])), (int(p[1]), int(p[3])), (255, 0, 0) , 2)
            q_image_hough = qimage2ndarray.array2qimage(image)
            self.ui.hough_output_image.setPixmap(QPixmap(q_image_hough).scaled(350,350))
            self.ui.hough_output_image.setScaledContents(True)

        elif self.ui.circles_check_box.isChecked():
            points_circle = hough_transform_circle(self.image_hough, float(self.ui.hough_rmin.text()), float(self.ui.hough_thresh.text()),
            float(self.ui.hough_step.text()))
            print(points_circle)
            for c in points_circle:
                image = cv2.circle(self.image_hough, (c[0], c[1]), c[2], (255, 0, 0))
            q_image_hough = qimage2ndarray.array2qimage(image)
            self.ui.hough_output_image.setPixmap(QPixmap(q_image_hough).scaled(500,500))
            self.ui.hough_output_image.setScaledContents(True)


    def open_image_hough(self):
        self.fileName_hough, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_hough:
            head, tail = os.path.split(self.fileName_hough)
            self.image_hough = cv2.cvtColor(cv2.imread(self.fileName_hough,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_hough.shape)
            self.q_image_hough = qimage2ndarray.array2qimage(self.image_hough)
            self.ui.hough_input_image.setPixmap(QPixmap(self.q_image_hough).scaled(500,500))
            self.ui.hough_input_image.setScaledContents(True)
            self.ui.hough_image_name.setText(tail)
            self.ui.hough_image_size.setText(str(self.image_hough.shape))


    def apply_contour(self):
        if self.fileName_snake and self.x_image_center and self.radius:
            image, contour = activeContour(self.fileName_snake, (self.x_image_center, self.y_image_center), self.radius, 
            int(self.ui.snake_alpha_input.text()), int(self.ui.snake_beta_input.text()), int(self.ui.snake_gamma_input.text()), 
            int(self.ui.snake_num_iter.text()))
            image = gray_rgb(image)
            for c in contour:
                cv2.circle(image, (c[0], c
                [1]), 2, (255, 0, 0), -1)
            q_image_snake = qimage2ndarray.array2qimage(image)
            self.ui.snake_input_image.setPixmap(QPixmap(q_image_snake).scaled(500,500))
            self.ui.snake_input_image.setScaledContents(True)
            print(contour)

    def clear_anchors(self):
        if self.fileName_snake:
            self.image_snake = cv2.cvtColor(cv2.imread(self.fileName_snake,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            self.ui.snake_input_image.setPixmap(QPixmap(self.q_image_snake).scaled(500,500))
            self.ui.snake_input_image.setScaledContents(True)
            self.flag_draw = True
            self.x_image_center = None
            self.y_image_center = None
            self.radius = None
            self.ui.snake_xy.setText("None")
            self.ui.snake_perimeter.setText("None")

    def draw_circle_on_image(self, event):
        if self.fileName_snake and self.counter == 0 and self.flag_draw:
            print(event.pos().x())
            self.x_clicked = event.pos().x()
            self.y_clicked = event.pos().y()
            xt = 500
            yt = 500
            self.x_image_center = self.x_clicked * (self.image_snake.shape[0] / xt)
            self.y_image_center = self.y_clicked * (self.image_snake.shape[0] / yt)
            self.x_image_center = math.floor(self.x_image_center)
            self.y_image_center = math.floor(self.y_image_center)
            self.counter +=1
            print(self.x_image_center, self.y_image_center)
        elif self.counter == 1 and self.flag_draw:
            self.x_clicked = event.pos().x()
            self.y_clicked = event.pos().y()
            xt = 500
            yt = 500
            self.x_image_second_point = self.x_clicked * (self.image_snake.shape[0] / xt)
            self.y_image_second_point = self.y_clicked * (self.image_snake.shape[0] / yt)
            self.x_image_second_point = math.floor(self.x_image_second_point)
            self.y_image_second_point = math.floor(self.y_image_second_point)
            self.radius = math.sqrt((self.x_image_center-self.x_image_second_point)**2 + (self.y_image_center-self.y_image_second_point)**2)
            self.counter = 0
            self.flag_draw = False
            print(self.x_image_second_point, self.y_image_second_point)
            print(self.radius)
            self.image_snake_1 = cv2.circle(self.image_snake,(self.x_image_center, self.y_image_center), int(self.radius), (255,0,0))
            q_image_snake = qimage2ndarray.array2qimage(self.image_snake_1)
            self.ui.snake_input_image.setPixmap(QPixmap(q_image_snake).scaled(500,500))
            self.ui.snake_input_image.setScaledContents(True)
            self.ui.snake_xy.setText("X= "+str(self.x_image_center)+", Y= "+str(self.y_image_center))
            self.ui.snake_perimeter.setText("Perimeter= "+str(self.radius))




    def open_snake_image(self):
        self.fileName_snake, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_snake:
            self.flag_draw = True
            head, tail = os.path.split(self.fileName_snake)
            self.image_snake = cv2.cvtColor(cv2.imread(self.fileName_snake,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_snake.shape)
            self.q_image_snake = qimage2ndarray.array2qimage(self.image_snake)
            self.ui.snake_input_image.setPixmap(QPixmap(self.q_image_snake).scaled(500,500))
            self.ui.snake_input_image.setScaledContents(True)
            self.ui.snake_image_name.setText(tail)
            self.ui.snake_image_size.setText(str(self.image_snake.shape))


    def apply_harris_variables(self):
        if str(self.ui.harris_mode_select.currentText()) == "Thresholding":
            self.ui.harris_param_input.setText("1000")
            self.ui.harris_param_input.setDisabled(False)
            self.ui.harris_range_text.setText("Range: 0-10000")
        if str(self.ui.harris_mode_select.currentText()) == "Non Maxima Supression":
            self.ui.harris_param_input.setDisabled(True)
            self.ui.harris_range_text.setText("No range")
        if str(self.ui.harris_mode_select.currentText()) == "Local Thresholding":
            self.ui.harris_param_input.setDisabled(True)
            self.ui.harris_range_text.setText("No range")

    def open_image_harriss(self):
        self.fileName_harris, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_harris:
            head, tail = os.path.split(self.fileName_harris)
            self.im_harris = Image.open(self.fileName_harris)
            self.image_harris = self.rgb2gray(cv2.cvtColor(cv2.imread(self.fileName_harris,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
            print(self.image_harris.shape)
            q_image_harris = qimage2ndarray.array2qimage(self.image_harris)
            self.ui.harris_input_image.setPixmap(QPixmap(q_image_harris).scaled(400,400))
            self.ui.harris_input_image.setScaledContents(True)
            self.ui.harris_image_name.setText(tail)
            self.ui.harris_image_size.setText(str(self.image_harris.shape))


    def apply_harris(self):
        if self.im_harris:
                print(str(self.ui.harris_window_size_input.text()), str(self.ui.harris_k_input.text()), str(self.ui.harris_param_input.text()))
                self.output_im_harris = get_harris_corner(self.image_harris, int(self.ui.harris_window_size_input.text()),
                 k= float(self.ui.harris_k_input.text()), method= str(self.ui.harris_mode_select.currentText()), param= float(self.ui.harris_param_input.text()))
                q_image_harris = qimage2ndarray.array2qimage(self.output_im_harris)
                self.ui.harris_image_output.setPixmap(QPixmap(q_image_harris).scaled(400,400))
                self.ui.harris_image_output.setScaledContents(True)

    def open_image_hybrid_1(self):
        self.fileName_hybrid_1, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_hybrid_1:
            head, tail = os.path.split(self.fileName_hybrid_1)
            self.im_hybrid_1 = Image.open(self.fileName_hybrid_1)
            self.image_hybrid_1 = self.rgb2gray(imread(self.fileName_hybrid_1))
            print(self.image_hybrid_1.shape)
            q_image_filter = qimage2ndarray.array2qimage(self.image_hybrid_1)
            self.ui.hybrid_input_image_1.setPixmap(QPixmap(q_image_filter).scaled(350,350))
            self.ui.filter_input_image.setScaledContents(True)
            self.ui.hybrid_image_1_name.setText(tail)
            self.ui.hybrid_image_1_size.setText(str(self.image_hybrid_1.shape))


    def open_image_hybrid_2(self):
        self.fileName_hybrid_2, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_hybrid_2:
            head, tail = os.path.split(self.fileName_hybrid_2)
            self.im_hybrid_2 = Image.open(self.fileName_hybrid_2)
            self.image_hybrid_2 = self.rgb2gray(imread(self.fileName_hybrid_2))
            print(self.image_hybrid_2.shape)
            q_image_filter = qimage2ndarray.array2qimage(self.image_hybrid_2)
            self.ui.hybrid_input_image_2.setPixmap(QPixmap(q_image_filter).scaled(350,350))
            #self.ui.filter_input_image.setScaledContents(True)
            self.ui.hybrid_image_2_name.setText(tail)
            self.ui.hybrid_image_2_size.setText(str(self.image_hybrid_2.shape))


    def open_image_filter(self):
        self.fileName_filter, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_filter:
            head, tail = os.path.split(self.fileName_filter)
            self.image_filter = self.rgb2gray(imread(self.fileName_filter))
            print(self.image_filter.shape)
            q_image_filter = qimage2ndarray.array2qimage(self.image_filter)
            self.ui.filter_input_image.setPixmap(QPixmap(q_image_filter).scaled(350,350))
            self.ui.filter_input_image.setScaledContents(True)
            self.ui.filter_image_name.setText(tail)
            self.ui.filter_image_size.setText(str(self.image_filter.shape))
            self.ui.noise_combo_box.setCurrentIndex(0)
            self.ui.filter_combo_box.setCurrentIndex(0)


    def open_image_hist(self):
        self.fileName_hist, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        self.draw_image_hist()

    def draw_image_hist(self):
        if self.fileName_hist:
            # Clear graph everytime I browse a picture
            self.ui.hist_input_hist.clear()
            self.ui.hist_output_hist.clear()

            # Displaying image name and size and the picture
            head, tail = os.path.split(self.fileName_hist)
            self.image_hist = self.get_image_channel(imread(self.fileName_hist))
            q_image_hist = qimage2ndarray.array2qimage(self.image_hist)
            self.ui.hist_input_image.setPixmap(QPixmap(q_image_hist).scaled(350,350))
            self.ui.hist_image_name.setText(tail)
            self.ui.hist_image_size.setText(str(self.image_hist.shape))

            # Displaying the histogram
            self.freq, self.hist = get_hist_1((self.image_hist.flatten()).astype(int), 256)
            self.ui.hist_input_hist.setBackground('w')
            bg1 = pg.BarGraphItem(x=self.hist, height=self.freq, width= 1,  brush='r')
            self.ui.hist_input_hist.addItem(bg1)
            print(str(self.ui.hist_combbox.currentText()))
            self.draw_output_hist()                


    def get_image_channel(self, img):
        copy_img = img.copy()
        if str(self.ui.selectrgb.currentText()) == "Grey Image":
            return self.rgb2gray(img)
        if str(self.ui.selectrgb.currentText()) == "R color":
            copy_img[:,:,1] = 0
            copy_img[:,:,2] = 0
            return copy_img
        if str(self.ui.selectrgb.currentText()) == "G color":
            copy_img[:,:,0] = 0
            copy_img[:,:,2] = 0
            return copy_img
        if str(self.ui.selectrgb.currentText()) == "B color":
            copy_img[:,:,0] = 0
            copy_img[:,:,1] = 0
            return copy_img


    def draw_output_hist(self):
            if str(self.ui.hist_combbox.currentText()) == "Equalization":
                self.ui.hist_output_hist.clear()
            # Displaying the equalized histogram and it's corresponding picture
                equalized_hist = equalize_histogram_1(self.freq, (self.image_hist.flatten()).astype(int))
                freq_equalized, hist_equalized = get_hist_1(equalized_hist, 256)
                self.ui.hist_output_hist.setBackground('w')
                bg2 = pg.BarGraphItem(x=hist_equalized, height=freq_equalized, width= 1,  brush='r')
                self.ui.hist_output_hist.addItem(bg2)
                q_image_hist_output = qimage2ndarray.array2qimage(np.reshape(equalized_hist, self.image_hist.shape))
                self.ui.hist_output_image.setPixmap(QPixmap(q_image_hist_output).scaled(350,350))
            if str(self.ui.hist_combbox.currentText()) == "Normalization":
                self.ui.hist_output_hist.clear()
                #normalized_img = NormalizeImage(self.image_hist.flatten())
                normalized_img = NormalizeImage(self.image_hist)
                freq_normalized, hist_normalized = get_hist_1((normalized_img.flatten()).astype(int), 256)
                self.ui.hist_output_hist.setBackground('w')
                bg2 = pg.BarGraphItem(x=hist_normalized, height=freq_normalized, width= 1,  brush='r')
                self.ui.hist_output_hist.addItem(bg2)
                #q_image_hist_output = qimage2ndarray.array2qimage(np.reshape(normalized_img, self.image_hist.shape))
                q_image_hist_output = qimage2ndarray.array2qimage(normalized_img)
                self.ui.hist_output_image.setPixmap(QPixmap(q_image_hist_output).scaled(350,350))
            if str(self.ui.hist_combbox.currentText()) == "Threshold":
                self.ui.hist_output_hist.clear()
                self.thresh_img = global_threshold(self.image_hist)
                self.thresh_freq, self.thres_hist = get_hist_1((self.thresh_img.flatten()).astype(int), 256)
                bg2 = pg.BarGraphItem(x=self.thres_hist, height=self.thresh_freq, width= 1,  brush='r')
                self.ui.hist_output_hist.addItem(bg2)
                q_image_hist_output = qimage2ndarray.array2qimage(self.thresh_img)
                self.ui.hist_output_image.setPixmap(QPixmap(q_image_hist_output).scaled(350,350))





    def make_hybrid(self):
        if self.fileName_hybrid_1 and self.fileName_hybrid_2:
            self.hybrid_image = hybrid(self.im_hybrid_1, self.im_hybrid_2)
            q_image_hybrid = qimage2ndarray.array2qimage(self.hybrid_image)
            self.ui.hybrid_output_image.setPixmap(QPixmap(q_image_hybrid))

    def apply_noise(self):
        if str(self.ui.noise_combo_box.currentText()) == "Gaussian":
             GN= gaussian_noise(self.image_filter, mu = 0.0, std = 1.0)    
             g_image_filter = qimage2ndarray.array2qimage(GN)
             self.ui.filter_noise_image.setPixmap(QPixmap(g_image_filter).scaled(350,350))
        if str(self.ui.noise_combo_box.currentText()) == "Uniform":
             UN= uniform_noise(self.image_filter)    
             u_image_filter = qimage2ndarray.array2qimage(UN)
             self.ui.filter_noise_image.setPixmap(QPixmap(u_image_filter).scaled(350,350)) 
        if str(self.ui.noise_combo_box.currentText()) == "Salt and Pepper":
             SPN= salt_n_pepper(self.image_filter, 70 , 30)    
             S_image_filter = qimage2ndarray.array2qimage(SPN)
             self.ui.filter_noise_image.setPixmap(QPixmap(S_image_filter).scaled(350,350))    

    def apply_filter(self):
        if str(self.ui.filter_combo_box.currentText()) == "Gaussian": 
            GNF=gaussian_filter(self.image_filter, sigma = 1.0, kernelSize = 3) 
            gf_image_filter = qimage2ndarray.array2qimage(GNF)
            self.ui.filter_filter_image.setPixmap(QPixmap(gf_image_filter).scaled(350,350))   
        if str(self.ui.filter_combo_box.currentText()) == "Sobel": 
            SF=sobel_edge(self.image_filter, threshold = 0, mode = 'gray') 
            sf_image_filter = qimage2ndarray.array2qimage(SF)
            self.ui.filter_filter_image.setPixmap(QPixmap(sf_image_filter).scaled(350,350))  
        if str(self.ui.filter_combo_box.currentText()) == "Median": 
            MF=median_filter(self.image_filter, kernelSize = 3) 
            mf_image_filter = qimage2ndarray.array2qimage(MF)
            self.ui.filter_filter_image.setPixmap(QPixmap(mf_image_filter).scaled(350,350))  
        if str(self.ui.filter_combo_box.currentText()) == "Roberts": 
            RF=roberts_edge(self.image_filter, threshold = 0, mode = 'gray') 
            rf_image_filter = qimage2ndarray.array2qimage(RF)
            self.ui.filter_filter_image.setPixmap(QPixmap(rf_image_filter).scaled(350,350)) 
        if str(self.ui.filter_combo_box.currentText()) == "Prewitt": 
            PF=prewitt_edge(self.image_filter, threshold = 0, mode = 'gray') 
            pf_image_filter = qimage2ndarray.array2qimage(PF)
            self.ui.filter_filter_image.setPixmap(QPixmap(pf_image_filter).scaled(350,350))
        if str(self.ui.filter_combo_box.currentText()) == "Average": 
            AF=average_filter(self.image_filter, kernelSize = 3) 
            af_image_filter = qimage2ndarray.array2qimage(AF)
            self.ui.filter_filter_image.setPixmap(QPixmap(af_image_filter).scaled(350,350))
        if str(self.ui.filter_combo_box.currentText()) == "Canny": 
            CF=canny_edge(self.image_filter,  sigma = 0.1, gaussSize = 3, filter = sobel_edge, minThresh = 5, maxThresh = 20)
            cf_image_filter = qimage2ndarray.array2qimage(CF)
            self.ui.filter_filter_image.setPixmap(QPixmap(cf_image_filter).scaled(350,350))

    def open_image_template_A(self):
        self.fileName_template_A, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_template_A:
            self.flag_draw = True
            head, tail = os.path.split(self.fileName_template_A)
            self.image_template_A = cv2.cvtColor(cv2.imread(self.fileName_template_A,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_template_A.shape)
            self.q_image_template_A = qimage2ndarray.array2qimage(self.image_template_A)
            self.ui.Template_input_image_A.setPixmap(QPixmap(self.q_image_template_A).scaled(500,500))
            self.ui.Template_input_image_A.setScaledContents(True)
            self.ui.template_image_name_A.setText(tail)
            self.ui.template_image_size_A.setText(str(self.image_template_A.shape))

    def open_image_template_B(self):
        self.fileName_template_B, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_template_B:
            self.flag_draw = True
            head, tail = os.path.split(self.fileName_template_B)
            self.image_template_B = cv2.cvtColor(cv2.imread(self.fileName_template_B,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_template_B.shape)
            self.q_image_template_B = qimage2ndarray.array2qimage(self.image_template_B)
            self.ui.Template_input_image_B.setPixmap(QPixmap(self.q_image_template_B).scaled(500,500))
            self.ui.Template_input_image_B.setScaledContents(True)
            self.ui.template_image_name_B.setText(tail)
            self.ui.template_image_size_B.setText(str(self.image_template_B.shape))


    def open_image_sift_A(self):
        self.fileName_sift_A, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_sift_A:
            self.flag_draw = True
            head, tail = os.path.split(self.fileName_sift_A)
            self.image_sift_A = cv2.cvtColor(cv2.imread(self.fileName_sift_A,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_sift_A.shape)
            self.q_image_sift_A = qimage2ndarray.array2qimage(self.image_sift_A)
            self.ui.sift_input_image_A.setPixmap(QPixmap(self.q_image_sift_A).scaled(500,500))
            self.ui.sift_input_image_A.setScaledContents(True)
            self.ui.sift_image_name_A.setText(tail)
            self.ui.sift_image_size_A.setText(str(self.image_sift_A.shape))

    def open_image_sift_B(self):
        self.fileName_sift_B, _ =  QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if self.fileName_sift_B:
            self.flag_draw = True
            head, tail = os.path.split(self.fileName_sift_B)
            self.image_sift_B = cv2.cvtColor(cv2.imread(self.fileName_sift_B,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            print(self.image_sift_B.shape)
            self.q_image_sift_B = qimage2ndarray.array2qimage(self.image_sift_B)
            self.ui.sift_input_image_B.setPixmap(QPixmap(self.q_image_sift_B).scaled(500,500))
            self.ui.sift_input_image_B.setScaledContents(True)
            self.ui.sift_image_name_B.setText(tail)
            self.ui.sift_image_size_B.setText(str(self.image_sift_B.shape))

    def apply_sift(self):
        sift = SIFT()
        sift.read_image(self.fileName_sift_A, self.fileName_sift_B)
        start_time = time()
        sift_image = sift.apply_sift()
        end_time = time()
        comp_time_ms = end_time-start_time
        self.q_image_sift_output = qimage2ndarray.array2qimage(sift_image)
        self.ui.sift_features_matching.setPixmap(QPixmap(self.q_image_sift_output).scaled(500,500))
        self.ui.sift_features_matching.setScaledContents(True)
        self.ui.sift_time_elapsed.setText("Elapsed Time is: "+str(comp_time_ms))


    def apply_template_method(self):
        self.image = self.rgb2gray(np.array(Image.open(self.fileName_template_A)))
        template = self.rgb2gray(np.array(Image.open(self.fileName_template_B))) 
        self.template = template
        if str(self.ui.template_mode_select.currentText()) == "Correlation":
            cm, self.idx, time= match_template_corr( self.image, template, numPeaks = int(self.ui.no_peaks_input.text()), thresh = None)
            print(cm.shape)
            cm = map_img(cm)
            color_cm = np.stack((cm,)*3, axis=-1)
            if len(self.idx) > 0:
                for i in range(len(self.idx)):
                    if i == 0:
                        self.c_method = cv2.circle(color_cm,(self.idx[i][1], self.idx[i][0]), 40, (255,0,0), thickness= 3) 
                    else:
                        self.c_method = cv2.circle(self.c_method,(self.idx[i][1], self.idx[i][0]), 40, (0,255,0), thickness= 3)
            else:
                self.cm_method = color_cm            
            c_method = qimage2ndarray.array2qimage(self.c_method)
            self.ui.Template_matching_space.setPixmap(QPixmap(c_method).scaled(500,500))
            self.ui.Template_matching_space.setScaledContents(True)
        if str(self.ui.template_mode_select.currentText()) == "Zero-mean":
            zmm, self.idx, time= match_template_corr_zmean( self.image, template, numPeaks = int(self.ui.no_peaks_input.text()), thresh = None)   
            zmm = map_img(zmm)
            color_zmm = np.stack((zmm,)*3, axis=-1)
            
            if len(self.idx) > 0:
                for i in range(len(self.idx)):
                    if i == 0:
                        self.z_method = cv2.circle(color_zmm,(self.idx[i][1], self.idx[i][0]), 40, (255,0,0), thickness= 3) 
                    else:
                        self.z_method = cv2.circle(self.z_method,(self.idx[i][1], self.idx[i][0]), 40, (0,255,0), thickness= 3)
            else:
                self.zmm_method = color_zmm        
            z_method = qimage2ndarray.array2qimage(self.z_method)
            self.ui.Template_matching_space.setPixmap(QPixmap(z_method).scaled(500,500)) 
            self.ui.Template_matching_space.setScaledContents(True)
        if str(self.ui.template_mode_select.currentText()) == "SSD":
            sm, self.idx, time= match_template_ssd( self.image, template , numPeaks = int(self.ui.no_peaks_input.text()), thresh = None)
            sm = map_img(sm)
            color_sm = np.stack((sm,)*3, axis=-1)

            if len(self.idx) > 0:
                for i in range(len(self.idx)):
                    print(i)
                    if i == 0:
                        self.s_method = cv2.circle(color_sm,(self.idx[i][1], self.idx[i][0]), 40, (255,0,0), thickness= 3) 
                    else:
                        self.s_method = cv2.circle(self.s_method,(self.idx[i][1], self.idx[i][0]), 40, (0,255,0), thickness= 3)
            else:
                self.s_method = color_sm

            s_method = qimage2ndarray.array2qimage(self.s_method)
            self.ui.Template_matching_space.setPixmap(QPixmap(s_method).scaled(500,500))
            self.ui.Template_matching_space.setScaledContents(True)
        if str(self.ui.template_mode_select.currentText()) == "Normalized cross correlation":
            nm, self.idx ,time= match_template_xcorr( self.image, template , numPeaks = int(self.ui.no_peaks_input.text()), thresh = None) 
            n_method = qimage2ndarray.array2qimage(map_img(nm))
            nm = map_img(nm)
            color_nm = np.stack((nm,)*3, axis=-1)
            if len(self.idx) > 0:
                for i in range(len(self.idx)):
                    if i == 0:
                        self.n_method = cv2.circle(color_nm,(self.idx[i][1], self.idx[i][0]), 40, (255,0,0), thickness= 3) 
                    else:
                        self.n_method = cv2.circle(self.n_method,(self.idx[i][1], self.idx[i][0]), 40, (0,255,0), thickness= 3)
            else:
                self.n_method = color_nm        
            print(nm.shape)
            n_method = qimage2ndarray.array2qimage(self.n_method)
            self.ui.Template_matching_space.setPixmap(QPixmap(n_method).scaled(500,500))    
            self.ui.Template_matching_space.setScaledContents(True)
        self.ui.template_time_elapsed.setText("Time (ms) Elapsed is: "+str(time))

    def apply_detected_patterns(self):
        self.apply_template_method()
        image = self.image.copy()
        image_gray = np.stack((image,)*3, axis=-1)
        htemp, wtemp = self.template.shape
        print(self.idx)

        if len(self.idx) > 0:
            for i in range(len(self.idx)):
                if i == 0:
                    self.image_detected_patterns = cv2.rectangle(image_gray,(int(self.idx[i][1]-wtemp/2), int(self.idx[i][0]+htemp/2)), (int(self.idx[i][1]+wtemp/2), int(self.idx[i][0]-htemp/2)), (255,0,0), thickness= 3) 
                else:
                    self.image_detected_patterns = cv2.rectangle(self.image_detected_patterns,(int(self.idx[i][1]-wtemp/2), int(self.idx[i][0]+htemp/2)), (int(self.idx[i][1]+wtemp/2), int(self.idx[i][0]-htemp/2)), (0,255,0), thickness= 3)
        else:
            self.image_detected_patterns = image_gray

        d_method = qimage2ndarray.array2qimage(self.image_detected_patterns)
        self.ui.Template_detected_patterns.setPixmap(QPixmap(d_method).scaled(500,500))    
        self.ui.Template_detected_patterns.setScaledContents(True)



         




    def rgb2gray(self, rgb_image):
        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = window()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()