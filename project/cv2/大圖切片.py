
import cv2
import numpy as np
def camar(img):
    cv2.imshow(str(img),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def red_b(img,totle):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range (0,2):
        x,y,w,h = totle[i]
        #print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    camar(img)
def cut_pic(img,i):   #圖片跟目前讀取的圖片
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
    #print('共有多少個輪廓',len(contours),'目前看的圖',i)
    filename=0
    #red_b(img,contours)
    totle = []#順序
    for s in range(0,len(contours)): #共兩大塊
        x,y,w,h = cv2.boundingRect(contours[s]) #輪廓xy 寬高
        if w*h>800: 
            totle.append([x,y,w,h])
            totle = sorted(totle, key=(lambda totle:totle[0]))
            #print("totle",totle)
    #red_b(img,totle)
    for i1 in range (0,len(totle)):
        x,y,w,h = totle[i1]
        crop_img = img[y:y+h, x:x+w]
        if i==0:
            crop_img = cv2.copyMakeBorder(crop_img,20,20,20,20,cv2.BORDER_CONSTANT,value=[0,0,0]) 
            cv2.imwrite('number//score//'+str(filename)+'.jpg',crop_img )
            filename+=1
        elif i==1:
            #print('number//score//'+str(s)+'.jpg')
            crop_img = cv2.copyMakeBorder(crop_img,30,30,30,30,cv2.BORDER_CONSTANT,value=[0,0,0]) #因為太小所以要要外擴
            cv2.imwrite('number//seat_number//'+str(filename)+'.jpg', crop_img)
            filename+=1 
    #print('pic_size_check',pic_size_check)
    return  filename
k=[] #照片的數量 #0為座號
#print("共有大塊",contours)
for i in range(0,2):
    img = cv2.imread("number//red_boder//"+str(i)+".jpg")
    img_totle=cut_pic(img,i)
    k.append(img_totle)