import imghdr
import os
import re

import cv2
import self as self

import predict

if __name__ == '__main__':
    self.predictor = predict.CardPredictor()
    self.predictor.train_svm()
    cnt = 0
    correct = 0
    for root, dirs, files in os.walk("carplate//val"):
        for filename in files:
            filepath = os.path.join(root, filename)
            if imghdr.what(filepath) != 'jpeg':
                continue
            cnt += 1
            print("cnt", cnt)
            img_bgr = predict.imreadex(filepath)
            # resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            # for resize_rate in resize_rates:
            #     print("resize_rate:", resize_rate)
            r, roi, color = self.predictor.predict(img_bgr, ch_cnt=cnt)
                # if r:
                #     break
            r = ''.join(r)
            if re.match(r, filename) is not None:
                correct += 1

    var = correct * 100 / cnt
    print("正确率:", var, "%")
    print("cnt:", cnt)
    print("correct:", correct)

