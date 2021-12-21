import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, Add, GlobalAveragePooling2D

class BasicBlock(tf.keras.Model):
    #expansion=1
    def __init__(self, batch_momentum, channel_in=64, channel_out=64):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(channel_in, kernel_size=(3, 3), padding='same')
        self.bn1 = BatchNormalization(momentum=batch_momentum)
        self.av1 = Activation('relu')
        self.conv2 = Conv2D(channel_out, kernel_size=(3, 3), padding='same')
        self.bn2 = BatchNormalization(momentum=batch_momentum)
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av2= Activation('relu')

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x : x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size = (1, 1), padding = 'same')
    
    @tf.function        
    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.av1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        #out = self.av2(out)
        shortcut = self.shortcut(x)
        out = self.add([out, shortcut])
        y = self.av2(out)
        return y


class Bottleneck(tf.keras.Model):
    def __init__(self, channel_in=64, channel_out=256):
        super(Bottleneck, self).__init__()
        
        channel = channel_out 
        self.conv1 = Conv2D(channel_in, kernel_size = (1, 1), padding='same')
        self.bn1 = BatchNormalization()
        self.av1 = Activation('relu')
        self.conv2 = Conv2D(channel_in, kernel_size = (3, 3), padding='same')
        self.bn2 = BatchNormalization()
        self.av2 = Activation('relu')
        self.conv3 = Conv2D(channel_out, kernel_size = (1, 1), padding='same')
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = Activation('relu')

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x : x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size = (1, 1), padding='same')

    @tf.function
    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.av1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.av2(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        shortcut = self.shortcut(x)
        out = self.add([out, shortcut])
        y = self.av3(out)
        return y    


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, batch_momentum, channel_in=64, channel_out=512):
        super(ResNet, self).__init__()
        
        #self.channel_in = channel_in
        #self.pool = pool#True
        #self.num_classes = num_classes
        
        # block 0
        self.conv1 = Conv2D(channel_in, kernel_size=3, strides=1, padding='same')
        self.bn1 = BatchNormalization(momentum=batch_momentum)
        self.ac1 = Activation('relu')
        self.pool1 = MaxPool2D(pool_size = (3, 3), strides=2, padding = 'same')
        
        # layer 1
        self.layer1 = self.makelayer(batch_momentum, channel_in, 64, block, num_blocks[0])
        # Layer 2
        self.layer2 = self.makelayer(batch_momentum, 64, 128, block, num_blocks[1])
        # layer 3
        self.layer3 = self.makelayer(batch_momentum, 128, 256, block, num_blocks[2])
        # layer 4
        self.layer4 = self.makelayer(batch_momentum, 256, channel_out, block, num_blocks[3])
        self.pool2 = GlobalAveragePooling2D()
        self.fc1 = Dense(512)
        self.bn2 = BatchNormalization(momentum=batch_momentum)
        self.ac2 = Activation('relu')
        self.obj = Dense(num_classes[0])
        self.pos = Dense(num_classes[1])
    
    def makelayer(self, batch_momentum, channel_in, channel_out, block, num_blocks):
        #return tf.keras.Sequential[ block(channel_in, channel_out) for _ in range(num_blocks)]
        layer = []
        for i in range(num_blocks):
            if(i==num_blocks-1):
                layer.append(block(batch_momentum, channel_in, channel_out))
            else:
                layer.append(block(batch_momentum, channel_in, channel_in))
        return layer

    @tf.function
    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.ac1(out)
        out = self.pool1(out)
        for layer in self.layer1:
            out = layer(out, training=training)
        for layer in self.layer2:
            out = layer(out, training=training)
        for layer in self.layer3:
            out = layer(out, training=training)
        for layer in self.layer4:
            out = layer(out, training=training)
        out = self.pool2(out)
        #out = self.pool3(out)
        out = Flatten()(out)
        out = self.fc1(out)
        out = self.bn2(out, training=training)
        acts = self.ac2(out)
        out_o = self.obj(acts) # dense
        out_p = self.pos(acts) # dense
        return Activation('softmax')(out_o), Activation('softmax')(out_p), acts #out_o

    def get_model(self, model):
        x = tf.keras.layers.Input(shape=(84,84,1))
        tmp = tf.keras.Model(inputs=x, outputs=model.call(x))
        return tmp

def ResNet18(num_classes, batch_momentum, channel_in=64, channel_out=512):
    num_blocks = [2, 2, 2, 2]
    model = ResNet(BasicBlock, num_blocks, num_classes, batch_momentum, channel_in, channel_out)
    return model


def main():
    num_classes=[9,9]
    #model = ResNet18(num_classes=num_classes)
    #print(model.summary())
    x = tf.keras.layers.Input(shape=(42,42,1), batch_size=128, name='layer_in')
    tmp_model = tf.keras.Model(inputs=x, outputs=model.call(x), name='subclassing_model')
    print(tmp_model.summary())
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    
if __name__ == '__main__':
    main()
