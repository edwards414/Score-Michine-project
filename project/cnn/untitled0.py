from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import cv2
import numpy as np

def predict(CNN,img):
    x_Test4D_normalize = picchangesize(img)
    predict1 = CNN.predict(x_Test4D_normalize)
    print("機率為",(max(predict1[0]))*100,"%")
    predicted = np.argmax(predict1,axis=1)
    print("數字為",predicted[0])
    return predicted[0]
    #totle_number_list.append(predicted[0])
def picchangesize(img):
    img2_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉成黑白
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

(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()
print('train data= ',len(x_train_image))
print('test data=', len(x_test_image))
x_train_image=x_train_image/255
x_test_image=x_test_image/255
x_train_image.reshape(60000,28,28,1)
x_test_image.reshape(10000,28,28,1)

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
print(model.summary())
model.compile(optimizer='adam',
           loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
           metrics=['acc'])
#train_history = model.fit(x_train_image,y_train_label,epochs=20,batch_size=300,validation_split=0.2,
                         
    #                          verbose=2,)#batch_size=200
#model.save_weights('my_model_weights.h5')
#prediction=model.predict(x_Test4D_normalize)
#import pandas as pd
#pd.crosstab(y_test_label,prediction,
    #        rownames=['label'],colnames=['predict'])
#import matplotlib.pyplot as plt
#def show_train_history(train_acc,test_acc):
 #   plt.plot(train_history.history[train_acc])
 #   plt.plot(train_history.history[test_acc])
 #   plt.title('Train History')
 #   plt.ylabel(train_acc)
 #   plt.xlabel(test_acc)
 #   plt.legend(['train', 'test'], loc='upper left')
 #   plt.show()
    
    
model.load_weights('my_model_weights.h5')
img = cv2.imread("number//seat_number//0.jpg")
#img = cv2.imread("number//score//2.jpg")
#cv2.imshow('frame',img)
#cv2.waitKey(2000)

n=predict(model,img)
print(n)


