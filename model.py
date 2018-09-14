from keras import layers
import pandas as pd
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D,Input, Dense
from keras.models import Model
from matplotlib.pyplot import imshow
import numpy as np
import tensorflow as tf

traindf = pd.read_csv("all/training.csv")
testdf =pd.read_csv("all/test.csv")

#using lenet-5
traindf['Image'] = traindf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
testdf['Image'] = testdf['Image'].apply(lambda im: np.fromstring(im, sep=' '))

# print(type(train))
# print(train.shape)
# print(test.shape)

traindf1 = traindf.(np.isfinite(df[:]))

missdf =traindf[traindf.isnull().any(axis=1)]

missX = np.vstack(missdf['Image'].values)/255
missX = missX.astype(np.float32)
missX = missX.reshape(X_train.shape[0], 96,96,1)
missY = missdf.drop('Image', axis=1).values


X_train = np.vstack(traindf1['Image'].values)/255
X_train = X_train.astype(np.float32)
X_train = X_train.reshape(X_train.shape[0], 96,96,1)

np.random.shuffle(X_train)

Y_train= traindf.drop('Image', axis=1).values
Y_train = (Y_train-48)/48
Y_train =Y_train.astype(np.float32)

def flipimages():
    X_flip= []
    X= tf.placeholder(tf.float32, shape =(96,96,1))
    img1 = tf.img.flip_left_right(X)
    with tf.session() as sess:
        sess.run(global_variables_initializer())
        for i in range(X_train.shape[0])
            flipimg = sess.run{X_train[i], feed={X: img}}
            flipimgy = - Y_train[i]
            np.append(X_train, flipimg , axis=0 )
            np.append(Y_train, flipimgy, axis=0) 


def lenetmodel(input_shape):
    X_input = Input(input_shape) # input_shape = (96,96)
    X = X_input
    
    X= Conv2D(6, (5,5), strides=(1,1),padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2,2))(X)
    X = Activation('relu')(X)
    
    X= Conv2D(16, (5,5), strides=(1,1), name= 'conv2')(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2,2))(X)
    X = Activation('relu')(X)
    
    X= Conv2D(16, (5,5), strides=(1,1), name= 'conv3')(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2,2))(X)
    X = Activation('relu')(X)    
    
    X= Flatten()(X)
    X = Dense(512 , input_dim=(2304,))(X)
    X = Activation('relu')(X)
    X =  Dense(120, input_dim=(512,))(X)
    X = Activation('relu')(X)
    X = Dense(30, input_dim=(120,))(X)
        
    model = Model(inputs= X_input, outputs= X, name='lenet-5' )
    return model;



model = lenetmodel((96,96,1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 100, batch_size = 128)








