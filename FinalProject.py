# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cv2
from PIL import Image

IMAGE_SIZE = 224

def LoadData():                                                 #載入要訓練的圖片
    data = []
    label = []
    num = 250
    path_cwd = "train/"
    for i in range(1, 3):
        path = path_cwd + 'train' + str(i)
        for number in range(num):
            path_full = path + '/' + str(number) +'.jpg'
            image = Image.open(path_full).convert('L')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            img = np.reshape(image, (1, IMAGE_SIZE*IMAGE_SIZE))
            data.extend(img)
        label.extend(np.ones(num, dtype=np.int) * i)
    data = np.reshape(data, (num*i, IMAGE_SIZE*IMAGE_SIZE))
    return np.matrix(data), np.matrix(label).T                  #回傳數據和標籤

def svm(trainDataSimplified, trainLabel, testDataSimplified):   #SVM分類器
    clf3 = SVC(C=2.0)                                           #C為分類數目
    clf3.fit(trainDataSimplified, trainLabel)
    return clf3.predict(testDataSimplified)

def knn(neighbor, traindata, trainlabel, testdata):             #KNN分類器
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(traindata, trainlabel)
    return neigh.predict(testdata)

Data, Label = LoadData()
pca = PCA(1, True, True)                                  #設置參數，保留方差100%，copy = True， whiten = True
trainDataS = pca.fit_transform(Data)                      #擬合併降維訓練數據
print(trainDataS)

color = (0, 255, 0)                                         #設定框框顏色為(綠色)
cap = cv2.VideoCapture(0)                                   #設定攝影機預設參數
cascade_path = "haarcascade_frontalface_alt.xml"            #人臉識別分類器的路徑

#執行識別人臉
while True:
    _, frame = cap.read()   #讀取每一幀畫面

    #圖像灰階化，降低計算的複雜度
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #使用人臉識別分類器
    cascade = cv2.CascadeClassifier(cascade_path)

    #利用分類器識別出哪一類的人
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            #獲取臉部圖像，提交給模型識別
            m = frame_gray[y - 10: y + h + 10, x - 10: x + w + 10]

            top, bottom, left, right = (0, 0, 0, 0)
            image = m
            #獲取圖像尺寸
            h, w = image.shape
            longest_edge = max(h, w)

            #計算短邊需要增加多上像素寬度使其與長邊等長
            if h < longest_edge:
                dh = longest_edge - h
                top = dh // 2
                bottom = dh - top
            elif w < longest_edge:
                dw = longest_edge - w
                left = dw // 2
                right = dw - left
            else:
                pass

            BLACK = [0]

            constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

            #調整圖片大小
            image = cv2.resize(constant, (IMAGE_SIZE, IMAGE_SIZE))
            img_test = np.reshape(image, (1, IMAGE_SIZE * IMAGE_SIZE))
            testDataS = pca.transform(img_test)                        # 降維測試數據
            result = svm(trainDataS, Label, testDataS)                 # 使用SVM進行分類
            #result = knn(10, trainDataS, Label, testDataS)               # 使用KNN進行分類
            faceID = result[0]

            if faceID == 1:
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                cv2.putText(frame,'B10556001',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            elif faceID == 2:
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                cv2.putText(frame,'B10556026',(x + 30, y + 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Video", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break

#釋放所有攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()