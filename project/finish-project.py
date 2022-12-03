from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
#-----------------
import pygsheets
import pandas as pd
#-----------------------------
import board
import neopixel
import pigpio
import RPi.GPIO as GPIO
from gpiozero import LED
import time
from multiprocessing import Process
# 将服务设置为开机启动
# sudo systemctl enable pigpiod
from rpi_ws281x import PixelStrip, Color
import argparse
from multiprocessing import Process
#--------------------------------
LED_COUNT = 32        # Number of LED pixels.
#LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS =230  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

TrackingPin = 22 #感測器1
TrackingPin2 = 9 #感測器2
motor=LED(4) #馬達
pin_servo = 27 #伺服馬達
pwm = pigpio.pi()
pwm.set_mode(pin_servo, pigpio.OUTPUT)
pwm.set_PWM_frequency(pin_servo, 50)  # 50Hz frequency
#LED-------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
args = parser.parse_args()
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
strip.begin()
#----------------
def destroy():
    pwm.set_PWM_dutycycle(pin_servo, 0)
    pwm.set_PWM_frequency(pin_servo, 0)
def setDirection(angle):
    duty = 500 + (angle / 180) * 2000
    pwm.set_servo_pulsewidth(pin_servo, duty)
    print("角度=", angle, "-> duty=", duty)
def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TrackingPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(TrackingPin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    motor.off()
    setDirection(100)
    colorfill(strip,Color(0,0,0))
    colorfill_down(strip,Color(0,255,0))
def loop(TrackingPin,motor):
    
    while True:
        if GPIO.input(TrackingPin) == GPIO.HIGH:
            colorfill_down(strip,Color(255,0,0))
            print ('目前偵測到1')
            #colorfill(strip,Color(0,255,255))
            motor.on()
            time.sleep(2)
            motor.off()
            loop2(TrackingPin2)
            break
def loop2(TrackingPin2):
    while True:
        if GPIO.input(TrackingPin2) == GPIO.HIGH:
            print ('目前偵測到2')
            colorfill(strip,Color(0,255,255))
            camar_take_pic()
            setDirection(130)
            time.sleep(3)
            setDirection(100)
            break
        else:  
            print("沒有偵測到2")  
            loop(TrackingPin,motor,xasd) #給錯誤跳出迴圈
#----------------
#燈光
def colorfill(strip,color):
    count=16
    for i in range(16):
        strip.setPixelColor(i+count, color)
        strip.show()
def colorfill_down(strip,color):
    for i in range(16):
        strip.setPixelColor(i, color)
        strip.show()
def wheel(pos):
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)
def rainbow( wait_ms=20, iterations=1):
    global strip
    while True:
        for j in range(256 * iterations):
            for i in range(16):
                strip.setPixelColor(i, wheel((i + j) & 255))
            strip.show()
            time.sleep(wait_ms / 1000.0)
        
#-----------------
#cv2.imshow

def camar_take_pic():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = rotate_img(frame)    
    cv2.imwrite('abc.jpg',frame)
    digital_recognition_and_updata_WEB('abc.jpg') #辨識到上傳
    print("已照相")
    cap.release()
def rotate_img(img): #圖片旋轉
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotate_img = cv2.warpAffine(img, M, (w, h))
    
    return rotate_img
#------------------------------------
gc = pygsheets.authorize(service_account_file='project.json')
survey_url = 'https://docs.google.com/spreadsheets/d/1Q8oc9Y7EACEgQzROZuCZDBGK2Z28XPk6cEC_6r5P5Hk/edit#gid=0'
sh = gc.open_by_url(survey_url)
ws = sh.worksheet_by_title('text')

def red_b(img,contours):
    for i in range (0,len(contours) ):
        x,y,w,h = cv2.boundingRect(contours[i])
        #print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
   # camar(img)
#def camar(img):
 #   cv2.imshow(str(img),img)
  #  cv2.waitKey(5000)
  #  cv2.destroyAllWindows()
    
#-----------------
def red_boder(img):
    w,h,c = img.shape #抓取圖片寬高
    x,y,contrast,brightness = 50,0,200,0 
    crop_img = img[y:y+h, x+50:x+w-20]  #縮小圖片
    output = crop_img * (contrast/127 + 1) - contrast + brightness # 轉換公式
    output = np.clip(output, 0, 255)#增加對比200%
    output = np.uint8(output)
    gray_img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) #把圖片轉成灰階
    th1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY, 11, 2) #二值化
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(th1,kernel,iterations = 1) #影像侵蝕
    ret, th1 = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY_INV) #黑白調換
    #camar(th1)
    img = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)#轉BGR
    contours,hierarchy = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找輪廓
    for i in range (0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        filename=0 #檔名
        totle=[]#計算xy順序
        for i in range(0,len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            if w*h>5000 and w>100:
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
def red_boder_data_open_and_deal(contours): #大區域切小塊
    k=[] #照片的數量 #0為座號
    #print("共有大塊",contours)
    for i in range(0,contours):
        img = cv2.imread("number//red_boder//"+str(i)+".jpg")
        img_totle=cut_pic(img,i)
        k.append(img_totle)
    return k 
opo = 0

def picchangesize(img):
    global opo
    img = img[:,:,0]
    ret, img = cv2.threshold(img,150,255, cv2.THRESH_BINARY_INV)#轉二值化
    cv2.imwrite('picture//'+str(opo)+'.jpg',img)
    opo+=1
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
    model.add(layers.Flatten())#轉一維陣列
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation='softmax')) #0-9輸出
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
    print('seat_number',seat_number,'score',score)
    #print(str(score))
    ws.update_value('E'+str(seat_number+1), str(score))
#score = 0 #分數#換成分數
#if pic_totle==2:
#    score += totle_number_list[0]*10
#    score += totle_number_list[1]*1
#print("分數",score)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#---------------------------
if __name__=="__main__":
    setup()  #把傳感器1跟2設定好
    #p = Process(target= rainbow)
    try :
        loop(TrackingPin,motor)
        print("結束")
    except:
        print("錯誤")
        colorfill(strip,Color(255,0,0))
        colorfill_down(strip,Color(255,0,0))
        setDirection(130) 
        time.sleep(3)
        setDirection(100)
    finally:
        colorfill(strip,Color(0,0,0))
        colorfill_down(strip,Color(0,0,0))
        
        
    
    
    
#---------------------------