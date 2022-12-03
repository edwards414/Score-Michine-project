from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
#-----------------
import pygsheets
import pandas as pd
gc = pygsheets.authorize(service_account_file='project.json')
survey_url = 'https://docs.google.com/spreadsheets/d/1Q8oc9Y7EACEgQzROZuCZDBGK2Z28XPk6cEC_6r5P5Hk/edit#gid=0'
sh = gc.open_by_url(survey_url)
ws = sh.worksheet_by_title('text')

def red_b(img,contours):
    for i in range (0,len(contours) ):
        x,y,w,h = cv2.boundingRect(contours[i])
        #print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    camar(img)
def camar(img):
    cv2.imshow(str(img),img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
#-----------------
def red_boder(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([133,55,145])
    high_hsv = np.array([180,255,178])
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
    camar(mask)
    #print(len(contours))
    filename=0 #檔名
    totle=[]#計算xy順序
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if w*h>4000:
            totle.append([x,y,w,h])
            totle = sorted(totle,key =(lambda totle:totle[1]))
            #red_b(img,contours[i])
            #print(totle)
    for i in range(0,len(totle)):
        x,y,w,h = totle[i]
        if i==0:
            crop_img = img[y+50:y+h-50, x+30:x+w-50]
        else:
            crop_img = img[y+5:y+h+5, x+10:x+w-10]
        cv2.imwrite("number//red_boder//"+str(filename)+'.jpg', crop_img)
        filename+=1 
    print("filename",filename)
    return filename

def cut_pic(img,i):  
    img = cv2.blur(img,(5,5))
    #camar(img)
    imgrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉灰階
    #camar(imgrey)
    if i==0:
        ret, th1 = cv2.threshold(imgrey,150,255, cv2.THRESH_BINARY_INV)#轉二值化
        camar(th1)
    elif i==1:
        ret, th1 = cv2.threshold(imgrey,130,255, cv2.THRESH_BINARY_INV)#轉二值化
        camar(th1)
    contours,hierarchy = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#只檢查外輪廓
    #print('共有多少個輪廓',len(contours),'目前看的圖',i)
    filename=0
    #red_b(img,contours)
    totle = []#順序
    for s in range(0,len(contours)): #共兩大塊
        x,y,w,h = cv2.boundingRect(contours[s]) #輪廓xy 寬高
        if w*h>1000: 
            totle.append([x,y,w,h])
            totle = sorted(totle, key=(lambda totle:totle[0] ))
            #print("totle",totle)
    for i1 in range (0,len(totle)):
        x,y,w,h = totle[i1]
        expansion = 10 #外擴截圖
        crop_img = th1[y-expansion:y+h+expansion, x-expansion:x+w+expansion]
        try :#找出最適合的截圖範圍
            while crop_img.size == 0:
                crop_img = th1[y-expansion:y+h+expansion, x-expansion:x+w+expansion]
                expansion-=1
                #print(expansion)
        except:
            crop_img = th1[y:y+h, x:x+w]
        if i==0:
            crop_img = cv2.copyMakeBorder(crop_img,20,20,20,20,cv2.BORDER_CONSTANT,value=[0,0,0]) 
            cv2.imwrite('number//score//'+str(filename)+'.jpg',crop_img)
            filename+=1
        elif i==1:
            #print('number//score//'+str(s)+'.jpg')
            crop_img = cv2.copyMakeBorder(crop_img,30,30,30,30,cv2.BORDER_CONSTANT,value=[0,0,0]) #因為太小所以要要外擴
            cv2.imwrite('number//seat_number//'+str(filename)+'.jpg', crop_img)
            filename+=1 
    #print('pic_size_check',pic_size_check)
    return  filename
def red_boder_data_open_and_deal(contours): #大區域切小塊
    k=[] #照片的數量 #0為座號
    #print("共有大塊",contours)
    for i in range(0,contours):
        img = cv2.imread("number//red_boder//"+str(i)+".jpg")
        img_totle=cut_pic(img,i)
        k.append(img_totle)
    return k
        
def picchangesize(img):
    img = img[:,:,0]
    ret, img = cv2.threshold(img,150,255, cv2.THRESH_BINARY_INV)#轉二值化
    img2_gray = img#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉成黑白
    img3 = 255 - img2_gray #黑白調換
    img3 = img3.astype('float32')
    img3_min = np.amin(img3)#0.0
    img4 = img3 - np.amin(img3) #圖片上像素的01
    img5 = 255 * img4 / (np.amax(img4)+1)
    kernel = np.ones((5,5),np.uint8)
    img6 = cv2.dilate(img5,kernel,iterations = 3)
    img7 = cv2.resize(img6,(28,28),1)
    img8 = img6.astype('uint8')
    x_test_image = np.reshape(img7,(1,28,28))
    x_Test4D = x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32') #改變形狀
    x_Test4D_normalize = ( x_Test4D / np.amax(x_test_image) ) #除以像素最大值
    return x_Test4D_normalize

def CNN_model():
    model = keras.Sequential(name="CNN") 
    model.add(layers.Conv2D(16,(5,5),activation='relu',padding='same',input_shape=(28,28,1))) #第一層輸入
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32,(5,5),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())#轉一微陣列
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation='softmax')) #0-9輸出
    #print(model.summary())
    model.compile(optimizer='adam',
               loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['acc'])
    return model


def predict(CNN,img):
    x_Test4D_normalize = picchangesize(img)
    predict1 = CNN.predict(x_Test4D_normalize)
    #print("機率為",(max(predict1[0]))*100,"%")
    predicted = np.argmax(predict1,axis=1)
    #print("數字為",predicted[0])
    return predicted[0]
    #totle_number_list.append(predicted[0])

def list2number(list_n): #list裡面的數字轉數字
    n=len(list_n)
    totle=0
    for i in range(0,n):
        totle+=list_n[i]*(10**(n-i-1))
    if totle > 100:
        totle=100
    return totle
def digital_recognition_and_updata_WEB(img):
    CNN = CNN_model()
    CNN.load_weights('my_model_weights.h5')
    img = cv2.imread(str(img))
    contours=red_boder(img) #處理紅框
    pic_totle=red_boder_data_open_and_deal(contours) #處理切割
    seat_number=[]
    score=[]
    for i in range(0,pic_totle[1]): #座號
        img = cv2.imread("number//seat_number//"+str(i)+".jpg")
        n=predict(CNN,img)
        seat_number.append(n)  
    #print("--------------------")
    for i in range(0,pic_totle[0]): #分數
        img = cv2.imread("number//score//"+str(i)+".jpg")
        n=predict(CNN,img)
        score.append(n)
    #print("seat_number", seat_number)
    #print("score", score)
    seat_number = list2number(seat_number)
    score = list2number(score)
    print(seat_number,score)
    #print(str(score))
    ws.update_value('E'+str(seat_number+1), str(score))
digital_recognition_and_updata_WEB('abc.jpg')
#score = 0 #分數#換成分數
#if pic_totle==2:
#    score += totle_number_list[0]*10
#    score += totle_number_list[1]*1
#print("分數",score)
#cv2.waitKey(0)
#cv2.destroyAllWindows()