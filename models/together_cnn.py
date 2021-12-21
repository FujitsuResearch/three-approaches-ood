import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization


class Network(tf.keras.Model):

    def __init__(self, num_classes, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)

        # conv layer1
        self.conv1 = Conv2D(20, 5, 1)#, padding='same')
        # conv layer2
        self.conv2 = Conv2D(50, 5, 1)#, padding='same')
        # test batchnorm
        #self.bn1 = BatchNormalization()
        # fc layer1
        self.fc1 = Dense(500)
        # object(shape) layer
        self.obj = Dense(num_classes[0])
        # position layer
        self.pos = Dense(num_classes[1])

    @tf.function
    def call(self, inputs, training=None):
        x = Activation('relu')(self.conv1(inputs))
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Activation('relu')(self.conv2(x))
        #x = MaxPooling2D(pool_size=(2,2))(x)
        x = MaxPooling2D(pool_size=(36,36))(x)
        x = Flatten()(x)
        out = Activation('relu')(self.fc1(x))
        x_o = self.obj(out)
        x_p = self.pos(out)
        
        return (Activation('softmax')(x_o), Activation('softmax')(x_p)), out, #(x_o, x_p)

    def get_model(self, model):
        x = tf.keras.layers.Input(shape=(84,84,1))
        tmp = tf.keras.Model(inputs=x, outputs=model.call(x))
        return tmp

    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1) # in channels out_channels kernel stride
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2_number = nn.Linear(500, num_classes[0])
        self.fc2_color = nn.Linear(500, num_classes[1])
        self.classes = num_classes
        if len(num_classes) == 3:
            self.fc2_loc = nn.Linear(500, num_classes[2])
        self.name = "simple_cnn"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 50)
        x = F.relu(self.fc1(x))
        num = self.fc2_number(x)
        col = self.fc2_color(x)
        if len(self.classes) == 3:
            loc = self.fc2_loc(x)
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1), F.log_softmax(loc, dim=1)
        else:
            return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1)
    """

def creat():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(84,84,1)  ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # コンパイル
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model



def main():
    model = Network(num_classes=(10,9))#creat()
    print('creat')

    print(model.get_model(model).summary())
    #print(model.summary())

    #model.save('./model') 


if __name__ == '__main__':
    main()
    
