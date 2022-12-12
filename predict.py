import turtle
import math
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

from tensorflow import keras

from CNN import cnn_predict
from skimage import io, transform

# import matplotlib.pyplot as plt

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 来自opencv的sample，用于svm训练
# 使用方向梯度直方图Histogram of Oriented Gradients （HOG）作为特征向量
# 对一个图像进行抗扭斜（deskew）处理，把歪了的图片摆正
def deskew(img):
    m = cv2.moments(img)  # 计算图像中的中心矩(最高到三阶)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    # 图像的平移，参数:输入图像、变换矩阵、变换后的大小
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
#  hot，用于从图片中抽取特征向量
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


# 不能保证包括所有省份
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]


# 绘制直方图 axis=0展示水平直方图 axis=1展示垂直直方图
def draw_hist(pt, axis=0):
    rows = len(pt)
    row = [i for i in range(rows)]
    if axis == 0:
        plt.barh(row, pt, color='black', height=1)
    else:
        print("i am here")
        plt.bar(row, pt, color='black', width=1)
    plt.show()


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")

    def accurate_place_color(self, card_img):
        pic_hight, pic_width = card_img.shape[:2]
        # 通过颜色识别区域
        lower_blue = np.array([100, 73, 46])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([11, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        lower_green = np.array([10, 43, 46])
        upper_green = np.array([99, 255, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 46])
        hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        img_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # img_hsv = cv2.bitwise_not(hsv, hsv, mask=mask_black)
        # 根据阈值找到对应颜色 并灰度化
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', img_hsv)
        kernel = np.ones((6, 6), np.uint8)  # 3
        ret, img_thresh = cv2.threshold(img_hsv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('thresh', img_hsv)
        img_edge2 = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('edge2', img_edge2)
        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 尺寸判断操作
        areaMax = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > areaMax:
                areaMax = cv2.contourArea(cnt)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= areaMax]
        print('len(contours) color', len(contours))
        ret_img = card_img
        for cnt in contours:
            print(cv2.contourArea(cnt))
            # 生成最小外接矩阵
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point
            point_limit(low_point)
            point_limit(heigth_point)
            point_limit(left_point)
            point_limit(right_point)
            ret_img = card_img[low_point[1]:heigth_point[1], left_point[0]:right_point[0]]
            # cv2.imshow("ret", ret_img)
            # cv2.waitKey(0)
        return ret_img

    def accurate_place_color_green(self, card_img):
        pic_hight, pic_width = card_img.shape[:2]
        # 通过颜色识别区域
        lower_blue = np.array([100, 73, 46])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([11, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        lower_green = np.array([10, 30, 46])
        upper_green = np.array([99, 255, 255])
        # lower_black = np.array([0, 0, 0])
        # upper_black = np.array([180, 255, 46])
        hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # img_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow)
        img_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_green)
        # 根据阈值找到对应颜色 并灰度化
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', img_hsv)
        kernel = np.ones((6, 6), np.uint8)  # 3
        ret, img_thresh = cv2.threshold(img_hsv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('thresh', img_hsv)
        img_edge2 = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('edge2', img_edge2)
        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 尺寸判断操作
        areaMax = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > areaMax:
                areaMax = cv2.contourArea(cnt)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= areaMax]
        print('len(contours) color', len(contours))
        ret_img = card_img
        for cnt in contours:
            print(cv2.contourArea(cnt))
            # 生成最小外接矩阵
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point
            point_limit(low_point)
            point_limit(heigth_point)
            point_limit(left_point)
            point_limit(right_point)
            ret_img = card_img[low_point[1]:heigth_point[1], left_point[0]:right_point[0]]
            # cv2.imshow("ret_green", ret_img)
            # cv2.waitKey(0)
        return ret_img

    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        # print("accurate_place", card_img_hsv.shape)
        # cv2.imshow('hsv1', card_img_hsv)
        # cv2.waitKey(0)
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    def predict(self, car_pic, resize_rate=1, ch_cnt=1):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            pic_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * pic_rate)), interpolation=cv2.INTER_LANCZOS4)

        if resize_rate != 1:
            img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_LANCZOS4)
            pic_hight, pic_width = img.shape[:2]

        print("h,w:", pic_hight, pic_width)
        blur = self.cfg["blur"]
        # 高斯去噪
        oldimg = img.copy()
        if blur > 0:
            img = cv2.GaussianBlur(img, (blur, blur), 0)  # 图片分辨率调整
        # cv2.imshow('blur', img)

        # black_pixels = np.where(
        #     (img[:, :, 0] == 0) & 
        #     (img[:, :, 1] == 0) & 
        #     (img[:, :, 2] == 0)
        # )
        # img[black_pixels] = [255, 255, 255]
        # cv2.imshow('img', img)

        # 通过颜色识别区域
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([11, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([99, 255, 255])
        # lower_black = np.array([0, 0, 0])
        # upper_black = np.array([180, 255, 46])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        # mask_black = cv2.inRange(hsv, lower_black, upper_black)
        img_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # img_hsv = cv2.bitwise_and(hsv, hsv ,mask=mask_green)
        # img_hsv = cv2.bitwise_not(hsv, hsv, mask=mask_black)
        # 根据阈值找到对应颜色 并灰度化
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('hsv', img_hsv)
        # cv2.waitKey(0)
        # 灰度化
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', img)
        # equ = cv2.equalizeHist(img)
        # img = np.hstack((img, equ))
        # 去掉图像中不会是车牌的区域
        kernel = np.ones((6, 6), np.uint8)
        # img_opening = cv2.morphologyEx(img_hsv, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('open', img_opening)
        # img_opening = cv2.addWeighted(img_hsv, 1, img_opening, -1, 0)
        # cv2.imshow('origin', img_opening)

        # 找到图像边缘
        # 二值化 Otsu 滤波
        # 第一个retVal（得到的阈值值），第二个就是阈值化后的图像。
        ret, img_thresh = cv2.threshold(img_hsv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('erzhihua', img_thresh)
        # sobel 边缘检测
        # img_edge = cv2.Canny(img_thresh, 100, 200)
        # cv2.imshow('edge', img_edge)
        # 使用开运算和闭运算让图像边缘成为一个整体
        # 4 19
        # kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
        # 先开操作消除 后闭操作连接
        img_edge1 = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('edge1', img_edge1)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('edge2', img_edge2)

        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        try:
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 尺寸判断操作
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
        print('len(contours)', len(contours))
        # 一一排除不是车牌的矩形区域
        car_contours = []
        line_img = oldimg.copy()
        for cnt in contours:
            # 生成最小外接矩阵
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height  # 长宽比 这里width是高，height是宽，width比height大，高比宽大
            print(wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 1.9 and wh_ratio < 5.5:
                car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                newimg = cv2.drawContours(line_img, [box], 0, (0, 0, 255), 2)
                # cv2.imshow("edge4", newimg)
                # cv2.waitKey(0)

        print(len(car_contours))

        print("精确定位")
        card_imgs = []
        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        # 只是进行了旋转矫正
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            # 貌似可扩大边缘优化
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

            box = cv2.boxPoints(rect)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point
            # 几何方法进行旋转矫正，不懂细节
            if low_point[0] >= heigth_point[0]:  # 正角度
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(new_right_point)
                point_limit(heigth_point)
                point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)
                # cv2.imshow("card_positive", card_img)
                # cv2.waitKey(0)
            elif low_point[0] < heigth_point[0]:  # 负角度
                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                point_limit(right_point)
                point_limit(heigth_point)
                point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)
                # cv2.imshow("card", card_img)
                # cv2.waitKey(0)

            # 判断是否偏斜，矫正偏斜的情况

        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            # card_imgs[card_index] = self.accurate_place_color(card_img)
            green = yello = blue = black = white = 0
            # cv2.imshow('hsv1', card_img)
            # cv2.waitKey(0)
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            card_img = card_imgs[card_index]
            # cv2.imshow('hsv2', card_img_hsv)
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num

            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 50 and S > 43:  # 图片分辨率调整
                        yello += 1
                    elif 10 < H <= 99 and S > 43:  # 图片分辨率调整
                        green += 1
                    elif 85 < H <= 155 and S > 34:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                        card_img_count -= 1
                    elif 0 < H < 180 and 0 < S < 30 and 221 < V < 225:
                        white += 1
            color = "no"

            limit1 = limit2 = 0
            if yello * 2 >= card_img_count:
                color = "yello"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif blue * 2.5 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏偏紫
            elif green * 4 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif black + white >= card_img_count * 0.7:  # TODO
                color = "bw"
            print(color)
            colors.append(color)
            print(blue, green, yello, black, white, card_img_count)
            # cv2.imshow("color", card_img)
            # cv2.waitKey(0)
            if limit1 == 0:
                continue
            # 以上为确定车牌颜色
            # 以下为根据车牌颜色再定位，缩小边缘非车牌边界
            if color == "green":
                card_imgs[card_index] = self.accurate_place_color_green(card_img)
            else:
                card_imgs[card_index] = self.accurate_place_color(card_img)
            # xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
            # if yl == yh and xl == xr:
            #     continue
            # need_accurate = False
            # if yl >= yh:
            #     yl = 0
            #     yh = row_num
            #     need_accurate = True
            # if xl >= xr:
            #     xl = 0
            #     xr = col_num
            #     need_accurate = True
            # cv2.imshow("before", card_img)
            # card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
            #                                                                                                yl - (
            #                                                                                                        yh - yl) // 4:yh,
            #                                                                                                xl:xr]
            # cv2.imshow("after", card_imgs[card_index])
            # if need_accurate:  # 可能x或y方向未缩小，需要再试一次
            #     card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            #     xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
            #     self.accurate_place_color(card_img)
            #     if yl == yh and xl == xr:
            #         continue
            #     if yl >= yh:
            #         yl = 0
            #         yh = row_num
            #     if xl >= xr:
            #         xl = 0
            #         xr = col_num
            # card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
            #                                                                                                yl - (
            #                                                                                                        yh - yl) // 4:yh,
            #                                                                                                xl:xr]
        # 以上为车牌定位
        # 以下为识别车牌中的字符
        predict_result = []
        roi = None
        card_color = None
        cnn = keras.models.load_model('cnn.h5')
        # print('正在启动中,请稍等...')
        # cnn_predict(cnn, [np.zeros((80, 240, 3))])
        # print("已启动,开始识别吧！")
        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):

                card_img = card_imgs[i]
                # cv2.imwrite(f'chinese/test.jpg', card_img)
                if card_img.size == 0:
                    continue
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # cv2.imshow(f'{i}', gray_img)
                # cv2.waitKey(0)
                # cv2.imwrite(f'chinese/1.jpg', gray_img)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cv2.imshow("2", gray_img)
                # cv2.imwrite(f'chinese/2.jpg', gray_img)
                # 查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                # draw_hist(x_histogram)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = find_waves(x_threshold, x_histogram)
                # print("wave_peaks", wave_peaks)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # if i==3:
                #     cv2.imshow("detect", gray_img)
                # cv2.imwrite(f'chinese/3.jpg', gray_img)
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                # gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                # if i == 3:
                #     draw_hist(y_histogram, axis=1)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

                wave_peaks = find_waves(y_threshold, y_histogram)

                # for wave in wave_peaks:
                #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
                # 车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                if len(wave_peaks) <= 2:
                    continue
                point = wave_peaks[2]
                if point[1] - point[0] < max_wave_dis / 3:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue
                part_cards = seperate_card(gray_img, wave_peaks)
                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card
                    # cv2.imshow("part", part_card_old)
                    # cv2.waitKey(0)
                    # w = abs(part_card.shape[1] - SZ)//2
                    w = part_card.shape[1] // 3
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # print(part_card_old.shape)
                    # cv2.imshow("part", part_card_old)
                    # cv2.waitKey(0)
                    # cv2.imwrite("u.jpg", part_card)
                    # part_card = deskew(part_card)
                    part_card = preprocess_hog([part_card])
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        # print("chinese_shape", resp)
                        charactor = provinces[int(resp[0]) - PROVINCE_START]
                        # cv2.imwrite(f'chinese/{provinces[int(resp[0]) - PROVINCE_START - 1]}_{ch_cnt}.jpg',
                        #             part_card_old)
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(int(resp[0]))
                    # print(charactor)
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 8:  # 1太细，认为是边缘
                            print(part_card_old.shape)
                            continue
                    # cv2.imwrite(f'chinese/{charactor}.jpg',
                    #             part_card_old)

                    predict_result.append(charactor)
                roi = card_img
                card_color = color
                # break


                # print("before reshape", card_img.shape)
                # cv2.imshow("before'", card_img)
                roi = card_img.copy()
                # resize图片大小 先将原本的---> (80,240,3)
                # card_img = transform.resize(card_img, (80,240))
                card_img = cv2.resize(card_img, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :,
                           :3]  # 直接resize为(240,80)
                # 查看reshape后的图片shape
                # print("after reshape", card_img.shape)
                # cv2.imshow("after'", card_img)
                Lic_pred = cnn_predict(cnn, [card_img])  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
                # print(Lic_pred)
                if len(Lic_pred) > 0 and Lic_pred[0] is not None:
                    predict_result = Lic_pred[0][1]
                else:
                    continue
                card_color = color
                break

        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    # cA019W2 偏斜的图片 沪D71603
    cur_dir = sys.path[0]
    r, roi, color = c.predict(
        "test/cA019W2.jpg")
    print(r)
    x = ''.join(r)
    print(x)
    print(cur_dir)
    turtle.done()
