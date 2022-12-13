import imghdr
import os
import re
import timeit

import cv2
import self as self

import predict

if __name__ == '__main__':
    start = timeit.default_timer()
    self.predictor = predict.CardPredictor()
    self.predictor.train_svm()
    cnt = 0
    correct = 0
    pic = 0
    for root, dirs, files in os.walk("carplate//train"):
        for filename in files:
            filepath = os.path.join(root, filename)
            if imghdr.what(filepath) != 'jpeg':
                continue
            cnt += 1
            # print("1:", timeit.default_timer())
            print("cnt", cnt)
            img_bgr = predict.imreadex(filepath)
            # resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
            # for resize_rate in resize_rates:
            #     print("resize_rate:", resize_rate)
            r, roi, color = self.predictor.predict(img_bgr)
                # if r:
                #     break
            # r, roi = self.predictor.predict_cnn(filepath)
            # print("2:", timeit.default_timer())
            r = ''.join(r)
            print("r:", r, "filename:", filename)
            if r != '' and re.match(r, filename) is not None:
                correct += 1
            if roi is not  None:
                pic+=1

    var = correct * 100 / pic
    end = timeit.default_timer()
    print("正确率:", var, "%")
    print("cnt:", cnt)
    print("correct:", correct)
    print("pic:", pic)
    print("time: ", end - start)

