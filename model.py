from keras import layers
import pandas as pd
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D,Input, Dense
from keras.models import Model
from matplotlib.pyplot import imshow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

traindf = pd.read_csv("data/training.csv")
testdf =pd.read_csv("data/test.csv")
# print(  type(traindf.iloc[0]['Image']))
# before 'Image' is stored as string, it is converted to matrix
traindf['Image'] = traindf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
testdf['Image'] = testdf['Image'].apply(lambda im: np.fromstring(im, sep=' '))
missdf =traindf[traindf.isnull().any(axis=1) ]
missX = np.vstack(missdf['Image'].values)/255
missX = missX.astype(np.float32)
missX = missX.reshape(missX.shape[0], 96,96,1)
missY = missdf.drop('Image', axis=1).values

traindf1 = traindf[traindf.isnull().any(axis=1)!=1 ]
print( type(traindf1.iloc[1]['Image'] ))

X_train = np.vstack(traindf1['Image'].values)/255
X_train = X_train.astype(np.float32)
X_train = X_train.reshape(X_train.shape[0], 96,96,1)

Y_train= traindf1.drop('Image', axis=1).values
Y_train = (Y_train-48)/48
Y_train =Y_train.astype(np.float32)

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
    model.summary()
    return model

model = lenetmodel((96,96,1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 10, batch_size = 128, shuffle= True)

model.save('intermediate_model.h5')

output = model.predict(missX, batch_size=128 )
print(output.shape)
missY = missY.astype(np.float32)
missY = (missY-48)/48
for i in range(missX.shape[0]):
    for j in range(30):
        if( pd.isnull(missY[i][j]) ):
            missY[i][j] = output[i][j]                

X_train = np.append(X_train, missX , axis=0 )
Y_train = np.append(Y_train, missY , axis=0 )

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 20, batch_size = 128, shuffle=True, validation_split=0.2)

model.save('my_model_final.h5')

# To directly load
# model = lenetmodel((96,96,1))
# model.load_weights('my_model_final.h5')

np.random.seed(19680801)
for i in range(6):       
    k = random.randint(0, X_train.shape[0]-1)
    print(k)
    image = np.round(X_train[k] * 255)
    plt.subplot(2,3,i+1)
    plt.imshow(image.reshape(96,96), cmap ='gray' )
    plt.scatter( Y_train[k][0::2] * 48 + 48, Y_train[k][1::2] * 48 + 48, marker='x', s=10, color = 'red')
plt.show()