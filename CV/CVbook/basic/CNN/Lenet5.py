from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

(x_train,y_train),(x_test,y_test) = load_data()
x_train,x_test = x_train/255.,x_test/255.

img_hight,img_width = x_train.shape[1:]
num_classes = 10
img_channels = 1

model = Sequential()

model.add(Conv2D(6,kernel_size=(5,5),padding='same',activation='relu',input_shape=(img_hight,img_width,img_channels)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer='sgd',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1)
]

model.fit(x_train,y_train,batch_size=32,epochs=80,validation_data=(x_test,y_test),callbacks=callbacks)

result = model.predict(x_test)

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(x_test[6])
print(np.argmax(result[6]))