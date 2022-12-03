import cv2
import numpy as np
def show_pic(img):
    cv2.imshow(str(img),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def rotate_img(img): #圖片旋轉
     (h, w, d) = img.shape # 讀取圖片大小
     center = (w // 2, h // 2) # 找到圖片中心
     # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
     M = cv2.getRotationMatrix2D(center, 90, 1.0)
     rotate_img = cv2.warpAffine(img, M, (w, h))
     return rotate_img 
img = cv2.imread("picture\\abc.jpg")
def red_boder(img):
    w,h,c = img.shape
    x,y,contrast,brightness = 50,0,200,0
    crop_img = img[y:y+h, x+50:x+w-20]  #縮小圖片
    crop_img  =rotate_img(crop_img) 
    output = crop_img * (contrast/127 + 1) - contrast + brightness # 轉換公式
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    gray_img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    th1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(th1,kernel,iterations = 1) #影像侵蝕
    ret, th1 = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY_INV)
    show_pic(th1)
    img = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR) #把原來的圖片蓋掉
    contours,hierarchy = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range (0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        filename=0 #檔名
        totle=[]#計算xy順序
        for i in range(0,len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            if w*h>5000:
                totle.append([x,y,w,h])
                totle = sorted(totle,key =(lambda totle:totle[1]))
                #print(totle)
        for i in range(0,len(totle)):
            x,y,w,h = totle[i]
            if i==0:
                crop_img = img[y+50:y+h-50,x+30:x+w-50]
            else:
                crop_img = img[y+8:y+h-8, x+10:x+w-10]
            cv2.imwrite("number//red_boder//"+str(filename)+'.jpg', crop_img)
            filename+=1 
    return filename
red_boder(img)
#red_b(crop_img,contours)