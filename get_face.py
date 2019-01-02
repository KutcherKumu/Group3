# -*- coding: utf-8 -*-
import cv2

def get_face(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()                                  #讀取每一幀數據
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #將當前圖像轉換成灰階
        #人臉檢測，1.2和3分别為圖片縮放比例和需要檢測的有效點
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:                      #大於0檢測到人臉
            for faceRect in faceRects:              #框出人臉
                x, y, w, h = faceRect
                #將當前畫面擷取並保存成圖片
                img_name = "%s/%d.jpg" % (path_name, num)
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                num += 1
                if num > (catch_pic_num):           #超過最大儲存量，結束。
                    break
                #畫框框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                #顯示當前捕捉的數量
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d/250' % (num),(x + 30, y + 30), font, 1, (0,0,255),4)
        if num > (catch_pic_num):                   #超過最大儲存量，結束。
            break
        #顯示視窗
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    #釋放所有攝影機並關閉所有視窗
    cap.release()
    cv2.destroyAllWindows()

get_face("get_face", 0, 249, "image")