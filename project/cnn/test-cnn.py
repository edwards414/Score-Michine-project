from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
img =  cv2.imread("number//0.jpg")
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
#cv2.imshow('Carmer1',img2_gray)
x_Test4D = x_test_image.reshape(x_test_image.shape[0],28,28,1).astype('float32') #改變形狀
x_Test4D_normalize = ( x_Test4D / np.amax(x_test_image) ) #除以像素最大值
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #把資料載入
#print(x_train.shape)
#plt.imshow(x_train[0])
#print(y_train[0])
#plt.show()

# 將 features (影像特徵值)，轉換為 4 維矩陣
# 將 features，以 reshape 轉為 6000 x 28 x 28 x 1 的 4 維矩陣
#----------------------------<正確>
x_train = x_train.reshape(x_train.shape[0],28,28,1)/255.0#(60000, 28, 28) 六萬張圖片每張28x28 灰階 改變資料型態
x_test= x_test.reshape(x_test.shape[0],28,28,1)/255.0#(10000, 28, 28)  一萬張圖片每張28x28 灰階

def CNN_model():
    CNN = keras.Sequential(name="CNN") #CNN捲機神經網路
    CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1))) #第一層輸入
    CNN.add(layers.MaxPooling2D((2,2)))
    CNN.add(layers.Conv2D(64,(3,3),activation='relu')) #第二層
    CNN.add(layers.MaxPooling2D((2,2)))
    CNN.add(layers.Flatten())#轉一微陣列
    CNN.add(layers.Dense(128,activation="relu"))
    CNN.add(layers.Dense(64,activation='relu'))
    CNN.add(layers.Dense(10,activation='softmax')) #0-9輸出
    CNN.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return CNN
CNN = CNN_model()
#CNN.summary()
checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)

# Train the model with the new callback
#CNN.fit(x_train, y_train,epochs=10,
        #validation_data=(x_test,y_test),
#callbacks=[cp_callback])  

CNN.load_weights(checkpoint_path)
#loss,acc = CNN.evaluate(x_train,y_train,verbose=2)
#print("model, accuracy: {:5.2f}%".format(100 * acc))
predict1 = CNN.predict(x_Test4D_normalize)
print((max(predict1[0]))*100,"%")
predicted = np.argmax(predict1,axis=1)
print(predicted[0])

