from tensorflow import keras #神經網路編輯模組
from tensorflow.keras import layers
import time
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
heigt = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width,heigt)
while True:
    ret, img_1 = cap.read()
    time_start = time.time() #開始計時
    
    #cv2.imshow('Carmer',img_1)
    #-----------------<圖片處理>
    #img =  cv2.imread('n4.jpg')
    img = img_1[0:480,80:560]
    img2_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉成黑白
    
    img3 = 255 - img2_gray #黑白調換
    
    img3 = img3.astype('float32')
    img3_min = np.amin(img3)#0.0
    img4 = img3 - np.amin(img3) #圖片上像素的01
    img5 = 255 * img4 / (np.amax(img4)+1)
    
    cv2.imshow('gray',img3)
    
    kernel = np.ones((5,5),np.uint8)
    img6 = cv2.dilate(img5,kernel,iterations = 3)
    img7 = cv2.resize(img6,(28,28),1)
    img8 = img6.astype('uint8')
    x_test_image = np.reshape(img7,(1,28,28))
    cv2.imshow('gray1',img)
   
    
    #cv2.imshow('Carmer1',img2_gray)
    x_Test4D = x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32') #改變形狀
    x_Test4D_normalize = ( x_Test4D / np.amax(x_test_image) ) #除以像素最大值
    #print("x_Test4D_normalize",x_Test4D_normalize.shape)
    #--------------------</圖片處理>
    #(x_Train, y_Train), (x_Test, y_Test) = tf.keras.datasets.mnist.load_data() #把資料載入
    #x_train = x_train/255.0 #正規化0-1間
    #x_test = x_test/255.0
    # 將 features (影像特徵值)，轉換為 4 維矩陣
    # 將 features，以 reshape 轉為 6000 x 28 x 28 x 1 的 4 維矩陣
    #----------------------------<正確>
    #x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')#(60000, 28, 28) 六萬張圖片每張28x28 灰階 改變資料型態
    #x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32') #(10000, 28, 28)  一萬張圖片每張28x28 灰階
    
    # 將 features 標準化
    #x_Train4D_normalize = x_Train4D / 255
    #x_Test4D_normalize = x_Test4D / 255
    
    # 以 Onehot Encoding 轉換 label
    #y_TrainOneHot = np_utils.to_categorical(y_Train)
    #y_TestOneHot = np_utils.to_categorical(y_Test)
    #-------------------------</正確>
    #print(x_train.shape) #(60000, 28, 28) 六萬張圖片每張28x28
    #----------------------------------------------
    CNN = keras.Sequential(name="CNN") #CNN捲機神經網路
    CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1))) #第一層輸入
    CNN.add(layers.MaxPooling2D((2,2)))
    
    CNN.add(layers.Conv2D(64,(3,3),activation='relu')) #第二層
    CNN.add(layers.MaxPooling2D((2,2)))
    
    
    CNN.add(layers.Flatten())#轉一微陣列
    CNN.add(layers.Dense(128,activation="relu"))
    CNN.add(layers.Dense(64,activation='relu'))
    CNN.add(layers.Dense(10,activation='softmax')) #0-9輸出
    #--------------------------------------------
    #CNN.compile(loss='categorical_crossentropy',
     #           optimizer='adam',metrics=['accuracy'])
    #CNN.fit(x=x_Train4D_normalize,
    #                        y=y_TrainOneHot,validation_split=0.2,
    #                       epochs=30, batch_size=300,verbose=2)
    #print(CNN.summary())
    predict1 = CNN.predict(x_Test4D_normalize)
    predicted = np.argmax(predict1,axis=1)
    #predicted =CNN.predict(x_Test4D_normalize)
    print(predicted)
    print(max(predict1[0]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time_end = time.time()    #結束計時q
    time_c= time_end - time_start   #執行所花時間
    print('time cost', time_c, 's')
cap.release()
cv2.destroyWindow()