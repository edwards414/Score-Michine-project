from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
# 匯入資料
from keras.datasets import mnist
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
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())#轉一微陣列
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax')) #0-9輸出
print(model.summary())
model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])
train_history = model.fit(x_train_image,y_train_label,epochs=20,batch_size=300,validation_split=0.2,
                          
                              verbose=2,)#batch_size=200
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel(test_acc)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')
model.save_weights('my_model_weights.h5') #權重儲存

