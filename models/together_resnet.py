import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, Add, GlobalAveragePooling2D

class BasicBlock(tf.keras.Model):
    def __init__(self, channel_in=64, channel_out=64, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(channel_out, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.av1 = Activation('relu')
        self.conv2 = Conv2D(channel_out, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out, stride)
        self.add = Add()
        self.av2= Activation('relu')

    def _shortcut(self, channel_in, channel_out, stride):
        if channel_in == channel_out:
            return lambda x : x
        else:
            return self._projection(channel_out, stride)

    def _projection(self, channel_out, stride):
        return Conv2D(channel_out, kernel_size=(1, 1), strides=stride, padding='same')
    
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

## TODO check strides
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
    def __init__(self, block, num_blocks, num_classes, channel_in=64, channel_out=512):
        super(ResNet, self).__init__()
        
        #self.channel_in = channel_in
        #self.pool = pool#True
        #self.num_classes = num_classes
        
        # block 0
        self.conv1 = Conv2D(channel_in, kernel_size=3, strides=1, padding='same')
        #self.conv1 = Conv2D(channel_in, kernel_size=7, strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding = 'same')
        
        # layer 1
        self.layer1 = self.makelayer(block, num_blocks[0], channel_in, 64, stride=1)
        # Layer 2
        self.layer2 = self.makelayer(block, num_blocks[1], 64, 128, stride=2)
        # layer 3
        self.layer3 = self.makelayer(block, num_blocks[2], 128, 256, stride=2)
        # layer 4
        self.layer4 = self.makelayer(block, num_blocks[3], 256, channel_out, stride=2)
        self.pool2 = GlobalAveragePooling2D()
        self.fc1 = Dense(512)
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')
        self.obj = Dense(num_classes[0])
        self.pos = Dense(num_classes[1])
    
    def makelayer(self, block, num_blocks, channel_in, channel_out, stride):
        #return tf.keras.Sequential[ block(channel_in, channel_out) for _ in range(num_blocks)]
        layer = []
        for i in range(num_blocks):
            if (i==0):
                layer.append(block(channel_in, channel_out, stride))
            #if(i==num_blocks-1):
                #layer.append(block(channel_in, channel_out))
            else:
                layer.append(block(channel_in, channel_out))
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
        return Activation('softmax')(out_o), Activation('softmax')(out_p), acts #tf.keras.utils.normalize(acts, axis=-1, order=2)

    def get_model(self, model):
        x = tf.keras.layers.Input(shape=(84,84,1))
        tmp = tf.keras.Model(inputs=x, outputs=model.call(x))
        return tmp

class ResNet_cifar(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, channel_in=16, channel_out=64):
        super(ResNet, self).__init__()
        
        # block 0
        self.conv1 = Conv2D(channel_in, kernel_size=3, strides=1, padding='same')
        self.bn1 = BatchNormalization()
        self.ac1 = Activation('relu')
        
        # layer 1
        self.layer1 = self.makelayer(block, num_blocks[0], channel_in, 16, stride=1)
        # Layer 2
        self.layer2 = self.makelayer(block, num_blocks[1], 16, 32, stride=2)
        # layer 3
        self.layer3 = self.makelayer(block, num_blocks[2], 32, 64, stride=2)
        # layer 4
        #self.layer4 = self.makelayer(block, num_blocks[3], 256, channel_out, stride=2)
        self.pool2 = GlobalAveragePooling2D()
        self.fc1 = Dense(512)
        self.bn2 = BatchNormalization()
        self.ac2 = Activation('relu')
        self.obj = Dense(num_classes[0])
        self.pos = Dense(num_classes[1])
    
    def makelayer(self, block, num_blocks, channel_in, channel_out, stride):
        #return tf.keras.Sequential[ block(channel_in, channel_out) for _ in range(num_blocks)]
        layer = []
        for i in range(num_blocks):
            if (i==0):
                layer.append(block(channel_in, channel_out, stride))
            #if(i==num_blocks-1):
                #layer.append(block(channel_in, channel_out))
            else:
                layer.append(block(channel_in, channel_out))
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
        #for layer in self.layer4:
            #out = layer(out, training=training)
        out = self.pool2(out)
        out = Flatten()(out)
        out = self.fc1(out)
        out = self.bn2(out, training=training)
        acts = self.ac2(out)
        out_o = self.obj(acts) # dense
        out_p = self.pos(acts) # dense
        return Activation('softmax')(out_o), Activation('softmax')(out_p), acts #tf.keras.utils.normalize(acts, axis=-1, order=2)

    def get_model(self, model):
        x = tf.keras.layers.Input(shape=(32,32,1))
        tmp = tf.keras.Model(inputs=x, outputs=model.call(x))
        return tmp

def ResNet18(num_classes, channel_in=64, channel_out=512):
    num_blocks = [2, 2, 2, 2]
    model = ResNet(BasicBlock, num_blocks, num_classes, channel_in, channel_out)
    return model

def ResNet32(num_classes, channel_in=32, channel_out=64):
    num_blocks = [5, 5, 5]
    model = ResNet_cifar(BasicBlock, num_blocks, num_classes, channel_in, channel_out)
    return model

def main():
    with tf.device("/cpu:0"):
        num_classes=[9,9]
        model = ResNet18(num_classes=num_classes)
        #basic = BasicBlock()
        #print(model.get_model(model).summary())
        def get_functional_model(model):
            x = tf.keras.layers.Input(shape=(42,42,1), batch_size=256, name='layer_in')
            temp_model = tf.keras.Model(inputs=[x],
                                        outputs=model.call(x),  # サブクラス化したモデルの`call`メソッドを指定
                                        name='subclassing_model')  # 仮モデルにも名前付け
            return temp_model
    
        temp_model = get_functional_model(model)
        # saving
        tf.keras.utils.plot_model(temp_model, to_file='model.png', dpi=256,
                                  show_shapes=True, show_layer_names=True,  # show shapes and layer name
                                  expand_nested=False)                      # will show nested block

    #print(model.summary())
    #print(model.get_model(model).summary())
    #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    ## how to subclassing tensorflow keras model plot_model
    #  apt install graphviz
    #  pip install pydot (pydot-ng) graphviz
    #  build subclassing model with eager option
    #  convert to Functional API get_functional_model() in this cord
    #  plot_model
    
    
if __name__ == '__main__':
    main()
