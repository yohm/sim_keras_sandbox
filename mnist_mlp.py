'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import sys,json
if len(sys.argv) != 2:
    sys.stderr.write("[Error] invalid number of arguments\n")
    sys.stderr.write("  Usage : python %s _input.json\n" % sys.argv[0])
    raise Exception("invalid argument")

with open( sys.argv[1] ) as f:
    params = json.load(f)

print(params)

import numpy as np
np.random.seed( params["_seed"] )

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense( params["dense1_size"], activation='relu', input_shape=(784,)))
model.add(Dropout( params["dropout1_prob"] ))
model.add(Dense( params["dense2_size"], activation='relu'))
model.add(Dropout( params["dropout2_prob"] ))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=0, verbose=1)


history = model.fit(x_train, y_train,
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping])
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
with open("_output.json", 'w') as f:
    obj = {"test_loss": score[0], "test_accuracy": score[1]}
    f.write( json.dumps(obj) )
    f.flush()

def plot_history(history):
    # print(history.history.keys())

    # plot history of accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.ylim(ymin = 0.9, ymax=1.0)
    plt.savefig("accuracy.png")
    plt.clf()
    #plt.show()

    # plot history of loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.ylim(ymin=0.0, ymax=0.25)
    plt.savefig("loss.png")
    #plt.show()

plot_history(history)

