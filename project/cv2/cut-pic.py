import cv2
import numpy as np
img = cv2.imread('100n6.png')
#imgrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#--
def red_boder():
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#轉HSV
    low_hsv = np.array([0,43,46]) #紅色數值
    high_hsv = np.array([10,255,255])
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv) #紅框變白框/黑底
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        crop_img = img[y+4:y+h-4, x+4:x+w-4]
        cv2.imwrite("number//red_boder//"+str(i)+'.jpg', crop_img)
    return len(contours)

def cut_pic(img,i):
    imgrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉灰階
    ret, th1 = cv2.threshold(imgrey,127, 255, cv2.THRESH_BINARY_INV)#轉二值化
    
    contours,hierarchy = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
    
    for i1 in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i1]) #輪廓xy 寬高
        crop_img = img[y:y+h, x:x+w]
        if i==0:
            cv2.imwrite("number//seat_number//"+str(i1)+'.jpg', crop_img)
        else:
            cv2.imwrite("number//score//"+str(i1)+'.jpg', crop_img)
            
def red_boder_data_open_and_deal(contours):
    for i in range(0,contours):
        img = cv2.imread("number//red_boder//"+str(i)+".jpg")
        cut_pic(img,i)

contours=red_boder() #處理紅框

red_boder_data_open_and_deal(contours) #處理切割

#ret, th1 = cv2.threshold(imgrey,127, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("th1",th1)
#contours,hierarchy = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
#for i in range(0,len(contours)):
#    x,y,w,h = cv2.boundingRect(contours[i]) #輪廓xy 寬高
#    cv2.rectangle(img,(x,y),(x+w,y+h),(0, 0 ,255),1)
#    crop_img = img[y-25:y+h+25, x-25:x+w+25]
#    cv2.imwrite("number//"+str(i)+'.jpg', crop_img)
#print(len(contours))
    #print("contours[",i,"]x=",x,"y=",y,"w=",w,"h=",h)
#cv2.imshow("img",mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
