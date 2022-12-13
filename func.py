import imghdr
import os
import re
import sys

import cv2
import numpy as np
from tensorflow import keras
from core import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict


class Detect:
    def __init__(self):
        cnt = 0
        cor = 0
        pic = 0
        self.unet = keras.models.load_model('unet.h5')
        self.cnn = keras.models.load_model('cnn.h5')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("开始识别")
        for root, dirs, files in os.walk("carplate//train"):
            for filename in files:
                filepath = os.path.join(root, filename)
                self.img_src_path = os.path.abspath(filepath)
                if imghdr.what(filepath) != 'jpeg':
                    continue
                cnt += 1
                img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
                h, w = img_src.shape[0], img_src.shape[1]
                if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                    lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
                    img_src_copy, Lic_img = img_src, [lic]
                else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                    img_src, img_mask = unet_predict(self.unet, self.img_src_path)
                    img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正

                Lic_pred = cnn_predict(self.cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
                if Lic_pred:
                    pic += 1
                    for i, lic_pred in enumerate(Lic_pred):
                        print("识别到的:", lic_pred[1], " 文件名:", filename)
                        if re.match(lic_pred[1], filename) is not None:
                            cor += 1
                else:  # Lic_pred为空说明未能识别
                    print("fail")
        print(" cnt:", cnt, " cor:", cor, " pic", pic)

    def closeEvent(self):  # 关闭前清除session(),防止'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    d = Detect()
