import numpy as np
import keras
from keras.layers import Activation,Dropout,Dense
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

(train_feature, train_target),(test_feature, test_target) = imdb.load_data(num_words=1000)

print(train_feature.shape,test_feature.shape)

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
train_feature = tokenizer.sequences_to_matrix(train_feature, mode='binary') #to convert data into binary form and create one hot encoding
test_feature = tokenizer.sequences_to_matrix(test_feature, mode='binary')
print(train_feature[0])

# One-hot encoding the output
num_classes = 2
train_target = keras.utils.to_categorical(train_target, num_classes)
test_target = keras.utils.to_categorical(test_target, num_classes)
print(train_target.shape)
print(test_target.shape)

model = Sequential()
model.add(Dense(1024,activation='relu',input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu',input_dim=1000))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_feature,train_target,epochs=10,batch_size=25,verbose=1)

# model.evaluate(train_feature,train_target)
print(model.evaluate(test_feature,test_target))