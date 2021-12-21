import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D


class Network(tf.keras.Model):

    def __init__(self, num_classes, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)

        # conv layer1
        self.conv1 = Conv2D(20, 5, 1)#, padding='same')
        self.ac1 = Activation('relu')
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        # conv layer2
        self.conv2 = Conv2D(50, 5, 1)#, padding='same')
        self.ac2 = Activation('relu')
        self.pool2 = GlobalAveragePooling2D()
        # flatten
        self.fl = Flatten()
        # fc layer
        self.fc1 = Dense(500)
        self.ac3 = Activation('relu')
        # output layer
        self.out = Dense(num_classes)
        # last activation
        self.ac4 = Activation('relu')

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.ac1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        x = self.pool2(x)
        x = self.fl(x)
        x = self.fc1(x)
        act = self.ac3(x)
        x_o = self.out(act)
            
        return self.ac4(x_o), act

                
    def get_model(self, model):
        x = tf.keras.layers.Input(shape=(84,84,1))
        tmp = tf.keras.Model(inputs=x, outputs=model.call(x))
        return tmp


def main():
    model = Network(num_classes=9)#creat()
    print('creat')

    print(model.get_model(model).summary())
    #print(model.summary())

    #model.save('./model') 


if __name__ == '__main__':
    main()
    
