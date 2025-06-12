import sys
import imutils
import cv2
import numpy as np
import dlib
from time import sleep
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)

        # Operasi Titik
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)

        # Operasi Histogram
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.equalHistogram)

        # Operasi Geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action_45_Derajat.triggered.connect(self.rotasi_45derajat)
        self.action40_Derajat.triggered.connect(self.rotasi45derajat)
        self.action_90_Derajat.triggered.connect(self.rotasi_90derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.actionTranspose.triggered.connect(self.transpose)
        self.action2X.triggered.connect(lambda: self.zoomIn(2))
        self.action3X.triggered.connect(lambda: self.zoomIn(3))
        self.action4X.triggered.connect(lambda: self.zoomIn(4))
        self.action1_2.triggered.connect(lambda: self.zoomOut(0.5))
        self.action1_4.triggered.connect(lambda: self.zoomOut(0.25))
        self.action3_4.triggered.connect(lambda: self.zoomOut(0.75))
        self.actionCrop.triggered.connect(self.crop)

        # Operasi Aritmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)

        # Operasi Boolean
        self.actionOperasi_AND.triggered.connect(self.booleanAND)
        self.actionOperasi_OR.triggered.connect(self.booleanOR)
        self.actionOperasi_XOR.triggered.connect(self.booleanXOR)

        # Operasi Spasial
        self.actionKernel1.triggered.connect(self.kernel1)
        self.actionKernel2.triggered.connect(self.kernel2)
        self.actionMean.triggered.connect(self.mean)
        self.actionMean_2x2.triggered.connect(self.mean2x2)
        self.actionGaussian.triggered.connect(self.gaussian)
        self.actionSharp_I.triggered.connect(self.sharpeningI)
        self.actionSharp_II.triggered.connect(self.sharpeningII)
        self.actionSharp_III.triggered.connect(self.sharpeningIII)
        self.actionSharp_IV.triggered.connect(self.sharpeningIV)
        self.actionSharp_V.triggered.connect(self.sharpeningV)
        self.actionSharp_VI.triggered.connect(self.sharpeningVI)
        self.actionSharp_Laplace.triggered.connect(self.sharpeningLaplace)
        self.actionMedian.triggered.connect(self.median)
        self.actionMax_Filter.triggered.connect(self.maxfilter)
        self.actionMin_Filter.triggered.connect(self.MinFilter)

        # Operasi Transformasi
        self.actionDFT_Smoothing_Image.triggered.connect(self.smoothimage)
        self.actionDFT_Smoothing_Image_Tepi.triggered.connect(self.DFTtepi)

        # Operasi Deteksi Tepi Citra
        self.actionDeteksi_Tepi.triggered.connect(self.Sobel)
        self.actionPrewitt.triggered.connect(self.Prewitt)
        self.actionRobert.triggered.connect(self.robert)
        self.actionCanny.triggered.connect(self.canny)

        # Morfologi
        self.actionDelasi.triggered.connect(self.delasi)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)
        self.actionErosi.triggered.connect(self.erosi)
        self.actionSkeletonizing.triggered.connect(self.skeletonizing)

        # Global Thresholding
        self.actionBinary.triggered.connect(self.binary)
        self.actionBinary_Inverse.triggered.connect(self.binarinvers)
        self.actionTrunc.triggered.connect(self.trunc)
        self.actionTo_Zero.triggered.connect(self.tozero)
        self.actionTo_Zero_Inverse.triggered.connect(self.invtozero)

        # Local Thresholding
        self.actionOtsu.triggered.connect(self.otsuT)
        self.actionMeanT.triggered.connect(self.meanT)
        self.actionGaussianT.triggered.connect(self.gaussianT)

        # Contour
        self.actionCountour.triggered.connect(self.countour)

        # Color Processing
        self.actionTracking.triggered.connect(self.colortrack)
        self.actionPicker.triggered.connect(self.Picker)

        # Object detection HaarCaseCade
        self.actionObject_Detection.triggered.connect(self.objectdetection)
        self.actionHistogram_Of_Gradien.triggered.connect(self.HOG)
        self.actionHistogram_Of_Gradien_Jalan.triggered.connect(self.HOGJalan)
        self.actionHaarCaseCade_Face_And_Eye.triggered.connect(self.FnE)
        self.actionHaarCaseCade_Pedestrian_Detection.triggered.connect(self.Pendestrian)
        self.actionCircle_Hough_Transform.triggered.connect(self.CircleHough)

        # Face Detection
        self.actionFacial_Landmark.triggered.connect(self.facialLandmark)
        self.actionSwap_Face.triggered.connect(self.swapFace)
        self.actionSwap_Face_Real_Time.triggered.connect(self.swapFaceRealTime)
        self.actionYawn_Detection.triggered.connect(self.yawnDetection)

    # A1 - A8
    # A1 - Menampilkan Image
    def fungsi(self):
        self.Image = cv2.imread("boeing.jpg")
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(1)

    # A2 - Menampilkan Citra
    def displayImage(self, label=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(
            self.Image,
            self.Image.shape[1],
            self.Image.shape[0],
            self.Image.strides[0],
            qformat,
        )

        img = img.rgbSwapped()

        if label == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
        else:
            self.label_2.setPixmap(QPixmap.fromImage(img))

    # A3 - Konversi Citra RGB ke Citra Grayscale
    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0]
                    + 0.587 * self.Image[i, j, 1]
                    + 0.114 * self.Image[i, j, 2],
                    0,
                    255,
                )

        self.Image = gray
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A4 - Pencerahan Citra
    def brightness(self):
        # agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)
                self.Image[i, j] = b

        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A5 - Pengaturan Kontras Citra
    def contrast(self):
        # agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)
                self.Image[i, j] = b

        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A6 Peregangan Kontras
    def contrastStretching(self):
        # agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255
                self.Image[i, j] = b

        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A7 - Negative Image
    def negative(self):
        # agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                pixel = self.Image.item(i, j)
                neg = 255 - pixel
                self.Image[i, j] = neg

        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A8 - Biner Image
    def biner(self):
        # agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                pixel = self.Image.item(i, j)
                if pixel == 180:
                    self.Image[i, j] = 0
                elif pixel < 180:
                    self.Image[i, j] = 1
                else:
                    self.Image[i, j] = 255

        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # A9 - C2
    # A9 - Histogram Citra Grayscale
    def grayHistogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0]
                    + 0.587 * self.Image[i, j, 1]
                    + 0.114 * self.Image[i, j, 2],
                    0,
                    255,
                )

        self.Image = gray
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 256, [0, 256])
        plt.title("Histogram Citra Grayscale")
        plt.show()

    # A10 - Histogram Citra RGB
    def rgbHistogram(self):
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title("Histogram Citra RGB")
        plt.show()

    # A11 - Histogram Equalization
    def equalHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.Image = cdf[self.Image]
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

        plt.plot(cdf_normalized, color="b")
        plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("CDF", "Histogram"), loc="upper left")
        plt.title("Histogram Equalization")
        plt.show()

    # B1 - Translasi Citra
    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # B2 - Rotasi Citra dan Transpose
    def rotasi_45derajat(self):
        self.rotasi(-45)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasi_90derajat(self):
        self.rotasi(-90)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - (w / 2)
        rotationMatrix[1, 2] += (nH / 2) - (h / 2)
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    def transpose(self):
        self.Image = np.transpose(self.Image)
        self.Image = cv2.resize(self.Image, (461, 211))
        self.displayImage(2)

    # B3 - Resize Citra
    def zoomIn(self, skala=2):
        resize_img = cv2.resize(
            self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC
        )
        cv2.imshow("Original", self.Image)
        cv2.imshow(f"Zoom In {skala}X", resize_img)
        cv2.waitKey()

    def zoomOut(self, skala=0.5):
        resize_img = cv2.resize(
            self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC
        )
        cv2.imshow("Original", self.Image)
        cv2.imshow(f"Zoom Out {skala}X", resize_img)
        cv2.waitKey()

    # B4 - Crop Image
    def crop(self):
        h, w = self.Image.shape[:2]
        start_row, start_col = int(h * 0.25), int(w * 0.25)
        end_row, end_col = int(h * 0.75), int(w * 0.75)
        crop = self.Image[start_row:end_row, start_col:end_col]
        cv2.imshow("Original", self.Image)
        cv2.imshow("Cropped", crop)
        cv2.waitKey()

    # C1 - Operasi Aritmatika
    def aritmatika(self):
        image1 = cv2.resize(cv2.imread("boeing.jpg", 0), (461, 211))
        image2 = cv2.resize(cv2.imread("traktor.jpg", 0), (461, 211))
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Tambah", image_tambah)
        cv2.imshow("Image Kurang", image_kurang)
        cv2.imshow("Image Kali", image_kali)
        cv2.imshow("Image Bagi", image_bagi)
        cv2.waitKey()

    # C2 - Operasi Boolean
    def booleanAND(self):
        image1 = cv2.resize(cv2.imread("boeing.jpg", 1), (461, 211))
        image2 = cv2.resize(cv2.imread("traktor.jpg", 1), (461, 211))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image AND", operasi)
        cv2.waitKey()

    def booleanOR(self):
        image1 = cv2.resize(cv2.imread("boeing.jpg", 1), (461, 211))
        image2 = cv2.resize(cv2.imread("traktor.jpg", 1), (461, 211))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image OR", operasi)
        cv2.waitKey()

    def booleanXOR(self):
        image1 = cv2.resize(cv2.imread("boeing.jpg", 1), (461, 211))
        image2 = cv2.resize(cv2.imread("traktor.jpg", 1), (461, 211))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image XOR", operasi)
        cv2.waitKey()

    # D1 - D6
    # D1 - Konvolusi 2D
    def conv(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]
        F_height = F.shape[0]
        F_width = F.shape[1]
        H = (F_height) // 2
        W = (F_width) // 2
        out = np.zeros((X_height, X_width))
        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += w * a
                out[i, j] = sum
        return out

    def conv2(self, X, F):
        X_Height = X.shape[0]
        X_Width = X.shape[1]

        F_Height = F.shape[0]
        F_Width = F.shape[1]

        H = 0
        W = 0

        batas = (F_Height) // 2

        out = np.zeros((X_Height, X_Width))

        for i in np.arange(H, X_Height - batas):
            for j in np.arange(W, X_Width - batas):
                sum = 0
                for k in np.arange(H, F_Height):
                    for l in np.arange(W, F_Width):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += w * a
                out[i, j] = sum
        return out

    def kernel1(self):
        # Kernel 3x3 dengan semua nilai 1
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Kernel 3x3")
        plt.show()

    def kernel2(self):
        # Kernel 3x3 dengan pola [6, 0, -6; 6, 1, -6; 6, 0, -6]
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[6, 0, -6], [6, 1, -6], [6, 0, -6]], dtype=np.float32)
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Kernel 3x3")
        plt.show()

    # D2 - Mean Filter
    def mean(self):
        # Mean filter 3x3
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Mean Filter 3x3")
        plt.show()

    def konvolusi2(self, X, kernel):
        height, width = X.shape[:2]
        out = np.zeros_like(X, dtype=np.float32)
        H, W = kernel.shape

        for i in range(H // 2, height - H // 2):
            for j in range(W // 2, width - W // 2):
                submatrix = X[i - H // 2 : i + H // 2 + 1, j - W // 2 : j + W // 2 + 1]
                mean_value = np.mean(submatrix)
                out[i, j] = mean_value
        return out

    def mean2x2(self):
        # Mean filter 2x2
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), dtype=np.float32) / 4
        hasil = self.konvolusi2(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Mean Filter 2x2")
        plt.show()

    # D3 - Gaussian Filter
    def gaussian(self):
        # Gaussian filter dengan kernel 5x5
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 345) * np.array(
            [
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 2],
            ],
            dtype=np.float32,
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Gaussian Filter")
        plt.show()

    # D4 - Image Sharpening
    def sharpeningI(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening I")
        plt.show()

    def sharpeningII(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening II")
        plt.show()

    def sharpeningIII(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening III")
        plt.show()

    def sharpeningIV(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1], [-2, 5, -2], [1, -2, 1]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening IV")
        plt.show()

    def sharpeningV(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening V")
        plt.show()

    def sharpeningVI(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening VI")
        plt.show()

    def sharpeningLaplace(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = (1.0 / 16) * np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )
        hasil = self.conv(img, kernel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Sharpening Laplace")
        plt.show()

    # D5 - Median Filter
    def median(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape

        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = [
                    img[i + k, j + l] for k in range(-3, 4) for l in range(-3, 4)
                ]
                neighbors.sort()
                hasil[i, j] = neighbors[24]  # median dari 49 elemen
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Median Filter")
        plt.show()

    # D6 - Max Filter & Min Filter
    def maxfilter(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape

        for i in range(3, h - 3):
            for j in range(3, w - 3):
                max_val = max(
                    [img[i + k, j + l] for k in range(-3, 4) for l in range(-3, 4)]
                )
                hasil[i, j] = max_val
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Max Filter")
        plt.show()

    def MinFilter(self):
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hasil = img.copy()
        h, w = img.shape

        for i in range(3, h - 3):
            for j in range(3, w - 3):
                min_val = min(
                    [img[i + k, j + l] for k in range(-3, 4) for l in range(-3, 4)]
                )
                hasil[i, j] = min_val
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Hasil Min Filter")
        plt.show()

    # E1-F2
    # E1- Discrete Fourier Transform
    def smoothimage(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += y.max()
        Img = np.array(
            [[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8
        )
        plt.imshow(Img)
        img = cv2.imread("noisy_image.png", 0)

        dft = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1
        )

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        Y, X = np.ogrid[:rows, :cols]
        mask[(Y - crow) ** 2 + (X - ccol) ** 2 <= r * r] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(
            cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1
        )
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap="gray")
        ax1.set_title("Input Image")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap="gray")
        ax2.set_title("FFT of Image")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap="gray")
        ax3.set_title("FFT with Mask")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap="gray")
        ax4.set_title("Inverse Image")
        plt.show()

    # E2- Discrete Fourier Transform
    def DFTtepi(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += y.max()

        img = np.array(
            [[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8
        )
        plt.imshow(img)
        img = cv2.imread("boeing.jpg", 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1
        )

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        Y, X = np.ogrid[:rows, :cols]
        mask[(Y - crow) ** 2 + (X - ccol) ** 2 <= r * r] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(
            cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1
        )
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap="gray")
        ax1.set_title("Input Image")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap="gray")
        ax2.set_title("FFT of Image")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap="gray")
        ax3.set_title("FFT with Mask")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap="gray")
        ax4.set_title("Inverse Fourier")
        plt.show()

    # F1- Deteksi Tepi
    def Sobel(self):
        if len(self.Image.shape) == 3:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.Image

        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        gx = self.conv(img, Sx)
        gy = self.conv(img, Sy)

        plt.imshow(gx, cmap="gray", interpolation="bicubic")
        plt.title("Sobel X")
        plt.axis("off")
        plt.show()
        plt.imshow(gy, cmap="gray", interpolation="bicubic")
        plt.title("Sobel Y")
        plt.axis("off")
        plt.show()

        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag / mag.max()) * 255
        mag = mag.astype(np.uint8)

        plt.imshow(mag, cmap="gray", interpolation="bicubic")
        plt.title("Operasi Sobel")
        plt.axis("off")
        plt.show()

        self.Image = mag
        self.displayImage(2)

    def Prewitt(self):
        if len(self.Image.shape) == 3:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.Image

        Sx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        Sy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        gx = self.conv(img, Sx)
        gy = self.conv(img, Sy)

        plt.imshow(gx, cmap="gray", interpolation="bicubic")
        plt.title("Prewitt X")
        plt.axis("off")
        plt.show()
        plt.imshow(gy, cmap="gray", interpolation="bicubic")
        plt.title("Prewitt Y")
        plt.axis("off")
        plt.show()

        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag / mag.max()) * 255
        mag = mag.astype(np.uint8)

        plt.imshow(mag, cmap="gray", interpolation="bicubic")
        plt.title("Operasi Prewitt")
        plt.axis("off")
        plt.show()

        self.Image = mag
        self.displayImage(2)

    def robert(self):
        if len(self.Image.shape) == 3:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.Image

        Sx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        Sy = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        gx = self.conv2(img, Sx)
        gy = self.conv2(img, Sy)

        plt.imshow(gx, cmap="gray", interpolation="bicubic")
        plt.title("Roberts X")
        plt.axis("off")
        plt.show()
        plt.imshow(gy, cmap="gray", interpolation="bicubic")
        plt.title("Roberts Y")
        plt.axis("off")
        plt.show()

        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag / mag.max()) * 255
        mag = mag.astype(np.uint8)

        plt.imshow(mag, cmap="gray", interpolation="bicubic")
        plt.title("Operasi Robert")
        plt.axis("off")
        plt.show()

        self.Image = mag
        self.displayImage(2)

    # F2- Canny Edge Detection
    def canny(self):
        if len(self.Image.shape) == 3:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.Image

        # 1. Noise reduction
        gauss = (1.0 / 57) * np.array(
            [
                [0, 1, 2, 1, 0],
                [1, 3, 5, 3, 1],
                [2, 5, 9, 5, 2],
                [1, 3, 5, 3, 1],
                [0, 1, 2, 1, 0],
            ],
            dtype=np.float32,
        )
        noise = self.conv(img, gauss).astype(np.uint8)

        # 2. Gradient (Sobel)
        Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        gx = self.conv(noise, Sx)
        gy = self.conv(noise, Sy)

        mag = np.sqrt(gx * gx + gy * gy)
        mag = (mag / mag.max()) * 255
        mag = mag.astype(np.uint8)

        theta = np.arctan2(gy, gx)
        angle = theta * 180.0 / np.pi
        angle[angle < 0] += 180

        H, W = img.shape
        Z = np.zeros((H, W), dtype=np.uint8)

        # 3. Non-maximum suppression
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q = 255
                r = 255
                # angle 0 45 90 135
                a = angle[i, j]
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                elif 22.5 <= a < 67.5:
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                elif 67.5 <= a < 112.5:
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                elif 112.5 <= a < 157.5:
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]
                if mag[i, j] >= q and mag[i, j] >= r:
                    Z[i, j] = mag[i, j]

        # 4. Hysteresis thresholding part 1
        weak, strong = 100, 150
        Zt = np.zeros_like(Z)
        for i in range(H):
            for j in range(W):
                v = Z[i, j]
                if v > weak:
                    Zt[i, j] = weak if v <= strong else 255
                else:
                    Zt[i, j] = 0

        # 5. Hysteresis thresholding part 2
        # eliminate weak edges not connected to strong
        Z_final = Zt.copy()
        strong_val = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if Z_final[i, j] == weak:
                    if any(
                        Z_final[n, m] == strong_val
                        for n, m in [
                            (i + 1, j - 1),
                            (i + 1, j),
                            (i + 1, j + 1),
                            (i, j - 1),
                            (i, j + 1),
                            (i - 1, j - 1),
                            (i - 1, j),
                            (i - 1, j + 1),
                        ]
                    ):
                        Z_final[i, j] = strong_val
                    else:
                        Z_final[i, j] = 0

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(noise, cmap="gray", interpolation="bicubic")
        axes[0, 0].set_title("1. Noise Reduction")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(mag, cmap="gray", interpolation="bicubic")
        axes[0, 1].set_title("2. Gradient Magnitude (Sobel)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(Z, cmap="gray", interpolation="bicubic")
        axes[0, 2].set_title("3. Non-Max Suppression")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(Zt, cmap="gray", interpolation="bicubic")
        axes[1, 0].set_title("4. Hysteresis Part 1")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(Z_final, cmap="gray", interpolation="bicubic")
        axes[1, 1].set_title("5. Hysteresis Part 2 (Final Output)")
        axes[1, 1].axis("off")

        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        self.Image = Z_final
        self.displayImage(2)

    # G1-H3
    # G1 - Morfologi Citra
    def delasi(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hasil = cv2.dilate(threshold, strel, iterations=1)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Dilasi")
        plt.show()

    def erosi(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hasil = cv2.erode(threshold, strel, iterations=1)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Erosi")
        plt.show()

    def opening(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hasil = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, strel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Opening")
        plt.show()

    def closing(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hasil = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, strel)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Closing")
        plt.show()

    def skeletonizing(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        ret, img = cv2.threshold(img, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        plt.imshow(skel, cmap="gray", interpolation="bicubic")
        plt.title("Skeletonizing")
        plt.show()

    # H1 - Global Thresholding
    def binary(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        t = 127
        maks = 255
        ret, thresh = cv2.threshold(img, t, maks, cv2.THRESH_BINARY)
        plt.imshow(thresh, cmap="gray", interpolation="bicubic")
        plt.title("Binary")
        plt.show()

    def binarinvers(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        t = 127
        maks = 255
        ret, thresh = cv2.threshold(img, t, maks, cv2.THRESH_BINARY_INV)
        plt.imshow(thresh, cmap="gray", interpolation="bicubic")
        plt.title("Binary Invers")
        plt.show()

    def trunc(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        t = 127
        maks = 255
        ret, thresh = cv2.threshold(img, t, maks, cv2.THRESH_TRUNC)
        plt.imshow(thresh, cmap="gray", interpolation="bicubic")
        plt.title("Trunc")
        plt.show()

    def tozero(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        t = 127
        maks = 255
        ret, thresh = cv2.threshold(img, t, maks, cv2.THRESH_TOZERO)
        plt.imshow(thresh, cmap="gray", interpolation="bicubic")
        plt.title("To Zero")
        plt.show()

    def invtozero(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        t = 127
        maks = 255
        ret, thresh = cv2.threshold(img, t, maks, cv2.THRESH_TOZERO_INV)
        plt.imshow(thresh, cmap="gray", interpolation="bicubic")
        plt.title("Inverse To Zero")
        plt.show()

    # H2 - Local/Adaptive Tresholding & Otsu Thresholding
    def meanT(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        print("piksel awal")
        print(img)
        hasil = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2
        )  # threshold(x,y) = mean(neighborhood of pixel(x,y)) - C
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Mean Thresholding")
        plt.show()

    def gaussianT(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        print("piksel awal")
        print(img)
        hasil = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2
        )  # threshold(x,y) = weighted_sum(neighborhood(x,y)) - C
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Gaussian Thresholding")
        plt.show()

    def otsuT(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        print(img)
        t = 130
        ret, hasil = cv2.threshold(img, t, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(hasil, cmap="gray", interpolation="bicubic")
        plt.title("Otsu Thresholding")
        plt.show()

    # H3 - Identifikasi Bentuk (Contour)
    def countour(self):
        if self.Image is None:
            print("Gambar belum dimuat.")
            return

        Img = self.Image.copy()
        gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True
            )
            cv2.drawContours(Img, [contour], 0, (255, 0, 0), 2)

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            shape = "Unknown"
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                (x1, y1, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            elif len(approx) == 8:
                shape = "Octagon"
            elif len(approx) > 8:
                shape = "Circle"

            cv2.putText(
                Img,
                shape,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        plt.imshow(Img, cmap="gray", interpolation="bicubic")
        plt.title("Contour")
        plt.show()

    # I1 - Color Tracking
    def colortrack(self):
        cam = cv2.VideoCapture(0)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Rentang warna default
            lower_color = np.array([25, 50, 50])
            upper_color = np.array([32, 255, 255])

            # Rentang warna untuk warna biru
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])

            # Rentang warna untuk warna merah
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            # Rentang warna untuk warna hijau
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])

            # Masking default
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Masking untuk warna biru
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

            # Masking untuk warna merah
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            result_red = cv2.bitwise_and(frame, frame, mask=mask_red)

            # Masking untuk warna hijau
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            result_green = cv2.bitwise_and(frame, frame, mask=mask_green)

            # Menampilkan hasil

            cv2.imshow("Blue Mask", mask_blue)
            cv2.imshow("Blue Result", result_blue)

            cv2.imshow("Red Mask", mask_red)
            cv2.imshow("Red Result", result_red)

            cv2.imshow("Green Mask", mask_green)
            cv2.imshow("Green Result", result_green)

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    # I2 - Color Picker
    def Picker(self):
        def nothing(x):
            pass

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Trackbar")

        cv2.createTrackbar("L-H", "Trackbar", 0, 179, nothing)
        cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbar", 179, 179, nothing)
        cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            L_H = cv2.getTrackbarPos("L-H", "Trackbar")
            L_S = cv2.getTrackbarPos("L-S", "Trackbar")
            L_V = cv2.getTrackbarPos("L-V", "Trackbar")
            U_H = cv2.getTrackbarPos("U-H", "Trackbar")
            U_S = cv2.getTrackbarPos("U-S", "Trackbar")
            U_V = cv2.getTrackbarPos("U-V", "Trackbar")

            lower_color = np.array([L_H, L_S, L_V])
            upper_color = np.array([U_H, U_S, U_V])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Hasil", result)

            key = cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    # I3 - I5
    # I3 - Object Detection
    def objectdetection(self):
        cam = cv2.VideoCapture("Mobil1.mp4")
        car_cascade = cv2.CascadeClassifier("haarcascade_car.xml")

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for x, y, w, h in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow("Video", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    # I4 - Histogram Of Gradient
    def HOG(self):
        image = data.astronaut()

        fd, hog_image = hog(
            image,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
            channel_axis=-1,
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        ax1.axis("off")
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title("Input image")

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis("off")
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title("Histogram of Oriented Gradients")
        plt.show()

    def HOGJalan(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        Photo = cv2.imread("Jalan.png")
        Photo = imutils.resize(Photo, width=min(400, Photo.shape[0]))

        (regions, _) = hog.detectMultiScale(
            Photo, winStride=(4, 4), padding=(4, 4), scale=1.05
        )

        for x, y, w, h in regions:
            cv2.rectangle(Photo, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("image", Photo)
        cv2.waitKey()

    # I5 - Haar Cascade Face and Eye Detection
    def FnE(self):
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

        Image = cv2.imread("Wajah.png")
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("No faces found")

        for x, y, w, h in faces:
            cv2.rectangle(Image, (x, y), (x + w, y + h), (127, 0, 255), 2)

            roi_gray = gray[y : y + h, x : x + w]
            roi_color = Image[y : y + h, x : x + w]

            eyes = eye_classifier.detectMultiScale(roi_gray)

            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        cv2.imshow("Face Detection", Image)
        cv2.waitKey(0)

    # I6 - I11
    # I6 - Haar Cascade-Pendestrian Detection
    def Pendestrian(self):
        body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
        cap = cv2.VideoCapture("Pedestrian.avi")
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                break

            frame = cv2.resize(
                frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

            for x, y, w, h in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow("Pedestrians", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    # I7 - Circle Hough Transform
    def CircleHough(self):
        img = cv2.imread("Circle.jpg", 0)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
        )
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow("detected circles", cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # I8 - Facial Landmark
    def facialLandmark(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def get_landmarks(im):
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces
            return np.matrix([[p.x, p.y] for p in predictor(gray, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(
                    im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255),
                )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        image = cv2.imread("trump.jpg")
        landmarks = get_landmarks(image)
        image_with_landmarks = annotate_landmarks(image, landmarks)

        cv2.imshow("Result", image_with_landmarks)
        cv2.imwrite("image_with_landmarks.jpg", image_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # I9 - Swap Face
    def swapFace(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        SCALE_FACTOR = 1
        FEATHER_AMOUNT = 11
        FACE_POINTS = list(range(17, 68))
        MOUTH_POINTS = list(range(48, 61))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        NOSE_POINTS = list(range(27, 35))
        JAW_POINTS = list(range(0, 17))

        ALIGN_POINTS = (
            LEFT_BROW_POINTS
            + RIGHT_EYE_POINTS
            + LEFT_EYE_POINTS
            + RIGHT_BROW_POINTS
            + NOSE_POINTS
            + MOUTH_POINTS
        )

        OVERLAY_POINTS = [
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS,
        ]

        COLOUR_CORRECT_BLUR_FRAC = 0.6

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(
                    im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255),
                )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def draw_convex_hull(im, points, color):
            points = cv2.convexHull(points)
            cv2.fillConvexPoly(im, points, color=color)

        def get_face_mask(im, landmarks):
            im = np.zeros(im.shape[:2], dtype=np.float64)
            for group in OVERLAY_POINTS:
                draw_convex_hull(im, landmarks[group], color=1)
            im = np.array([im, im, im]).transpose((1, 2, 0))
            im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
            im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
            return im

        def transformation_from_points(points1, points2):
            points1 = points1.astype(np.float64)
            points2 = points2.astype(np.float64)
            c1 = np.mean(points1, axis=0)
            c2 = np.mean(points2, axis=0)
            points1 -= c1
            points2 -= c2
            s1 = np.std(points1)
            s2 = np.std(points2)
            points1 /= s1
            points2 /= s2
            U, S, Vt = np.linalg.svd(points1.T * points2)
            R = (U * Vt).T
            return np.vstack(
                [
                    np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                    np.matrix([0.0, 0.0, 1.0]),
                ]
            )

        def read_im_and_landmarks(image):
            im = image
            im = cv2.resize(im, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            im = cv2.resize(
                im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR)
            )
            s = get_landmarks(im)
            return im, s

        def warp_im(im, M, dshape):
            output_im = np.zeros(dshape, dtype=im.dtype)
            cv2.warpAffine(
                im,
                M[:2],
                (dshape[1], dshape[0]),
                dst=output_im,
                borderMode=cv2.BORDER_TRANSPARENT,
                flags=cv2.WARP_INVERSE_MAP,
            )
            return output_im

        def correct_colours(im1, im2, landmarks1):
            blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
                - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
            )
            blur_amount = int(blur_amount)
            if blur_amount % 2 == 0:
                blur_amount += 1
            im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
            im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
            im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
            return (
                im2.astype(np.float64)
                * im1_blur.astype(np.float64)
                / im2_blur.astype(np.float64)
            )

        def swappy(image1, image2):
            im1, landmarks1 = read_im_and_landmarks(image1)
            im2, landmarks2 = read_im_and_landmarks(image2)
            M = transformation_from_points(
                landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS]
            )
            mask = get_face_mask(im2, landmarks2)
            warped_mask = warp_im(mask, M, im1.shape)
            combined_mask = np.max(
                [get_face_mask(im1, landmarks1), warped_mask], axis=0
            )
            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
            output_im = (
                im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
            )
            cv2.imwrite("output.jpg", output_im)
            image = cv2.imread("output.jpg")
            return image

        image1 = cv2.imread("trump.jpg")
        image2 = cv2.imread("potrait.jpeg")
        cv2.imshow("Face Swap Ori 1", image1)
        cv2.imshow("Face Swap Ori 2", image2)
        swapped = swappy(image1, image2)
        cv2.imshow("Face Swap 1", swapped)
        swapped = swappy(image2, image1)
        cv2.imshow("Face Swap 2", swapped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # I10 - Swap Face Real Time
    def swapFaceRealTime(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        SCALE_FACTOR = 1
        FEATHER_AMOUNT = 11
        FACE_POINTS = list(range(17, 68))
        MOUTH_POINTS = list(range(48, 61))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        NOSE_POINTS = list(range(27, 35))
        JAW_POINTS = list(range(0, 17))

        ALIGN_POINTS = (
            LEFT_BROW_POINTS
            + RIGHT_EYE_POINTS
            + LEFT_EYE_POINTS
            + RIGHT_BROW_POINTS
            + NOSE_POINTS
            + MOUTH_POINTS
        )

        OVERLAY_POINTS = [
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS,
        ]

        COLOUR_CORRECT_BLUR_FRAC = 0.6
        cascade_path = "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        dlibOn = True

        def get_landmarks(im, dlibOn):
            if dlibOn == True:
                rects = detector(im, 1)
                if len(rects) > 1:
                    return "error"
                if len(rects) == 0:
                    return "error"
                return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
            else:
                rects = cascade.detectMultiScale(im, 1.3, 5)
                if len(rects) > 1:
                    return "error"
                if len(rects) == 0:
                    return "error"
                x, y, w, h = rects[0]
                rect = dlib.rectangle(x, y, x + w, y + h)
                return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(
                    im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255),
                )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def draw_convex_hull(im, points, color):
            points = cv2.convexHull(points)
            cv2.fillConvexPoly(im, points, color=color)

        def get_face_mask(im, landmarks):
            im = np.zeros(im.shape[:2], dtype=np.float64)
            for group in OVERLAY_POINTS:
                draw_convex_hull(im, landmarks[group], color=1)
            im = np.array([im, im, im]).transpose((1, 2, 0))
            im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
            im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
            return im

        def transformation_from_points(points1, points2):
            points1 = points1.astype(np.float64)
            points2 = points2.astype(np.float64)
            c1 = np.mean(points1, axis=0)
            c2 = np.mean(points2, axis=0)
            points1 -= c1
            points2 -= c2
            s1 = np.std(points1)
            s2 = np.std(points2)
            points1 /= s1
            points2 /= s2
            U, S, Vt = np.linalg.svd(points1.T * points2)
            R = (U * Vt).T
            return np.vstack(
                [
                    np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                    np.matrix([0.0, 0.0, 1.0]),
                ]
            )

        def read_im_and_landmarks(fname):
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = cv2.resize(im, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_LINEAR)
            im = cv2.resize(
                im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR)
            )
            s = get_landmarks(im, dlibOn)
            return im, s

        def warp_im(im, M, dshape):
            output_im = np.zeros(dshape, dtype=im.dtype)
            cv2.warpAffine(
                im,
                M[:2],
                (dshape[1], dshape[0]),
                dst=output_im,
                borderMode=cv2.BORDER_TRANSPARENT,
                flags=cv2.WARP_INVERSE_MAP,
            )
            return output_im

        def correct_colours(im1, im2, landmarks1):
            blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
                - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
            )
            blur_amount = int(blur_amount)
            if blur_amount % 2 == 0:
                blur_amount += 1
            im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
            im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

            im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
            return (
                im2.astype(np.float64)
                * im1_blur.astype(np.float64)
                / im2_blur.astype(np.float64)
            )

        def face_swap(img, name):
            s = get_landmarks(img, True)
            if isinstance(s, str) and s == "error":
                print("No or too many faces")
                return img

            im1, landmarks1 = img, s
            im2, landmarks2 = read_im_and_landmarks(name)
            M = transformation_from_points(
                landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS]
            )
            mask = get_face_mask(im2, landmarks2)
            warped_mask = warp_im(mask, M, im1.shape)
            combined_mask = np.max(
                [get_face_mask(im1, landmarks1), warped_mask], axis=0
            )
            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
            output_im = (
                im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
            )

            cv2.imwrite("output.jpg", output_im)
            image = cv2.imread("output.jpg")
            frame = cv2.resize(
                image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR
            )
            return image

        cap = cv2.VideoCapture(0)
        filter_image = "trump.jpg"
        dlibOn = False
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            frame = cv2.resize(
                frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR
            )

            frame = cv2.flip(frame, 1)
            swapped = face_swap(frame, filter_image)
            cv2.imshow("Our Amazing Face Swapper", swapped)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    # I11 - Yawn Detection
    def yawnDetection(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                return "error"
            if len(rects) == 0:
                return "error"
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(
                    im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255),
                )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def top_lip(landmarks):
            top_lip_pts = []
            for i in range(50, 53):
                top_lip_pts.append(landmarks[i])
            for i in range(61, 64):
                top_lip_pts.append(landmarks[i])
            top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
            top_lip_mean = np.mean(top_lip_pts, axis=0)
            return int(top_lip_mean[:, 1])

        def bottom_lip(landmarks):
            bottom_lip_pts = []
            for i in range(65, 68):
                bottom_lip_pts.append(landmarks[i])
            for i in range(56, 59):
                bottom_lip_pts.append(landmarks[i])
            bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
            bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
            return int(bottom_lip_mean[:, 1])

        def mouth_open(image):
            landmarks = get_landmarks(image)
            if isinstance(landmarks, str) and landmarks == "error":
                return image, 0

            image_with_landmarks = annotate_landmarks(image, landmarks)
            top_lip_center = top_lip(landmarks)
            bottom_lip_center = bottom_lip(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        cap = cv2.VideoCapture(0)
        yawns = 0
        yawn_status = False
        while True:
            ret, frame = cap.read()
            image_landmarks, lip_distance = mouth_open(frame)
            prev_yawn_status = yawn_status
            if lip_distance > 25:
                yawn_status = True
                cv2.putText(
                    frame,
                    "Subject is Yawning",
                    (50, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                output_text = " Yawn Count: " + str(yawns + 1)
                cv2.putText(
                    frame,
                    output_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 127),
                    2,
                )
            else:
                yawn_status = False

            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

            cv2.imshow("Live Landmarks", image_landmarks)
            cv2.imshow("Yawn Detection", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle("Pengolahan Citra Digital")
window.show()
sys.exit(app.exec_())
