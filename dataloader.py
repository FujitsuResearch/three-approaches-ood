import numpy as np
import tensorflow as tf

def preprocess(x):
    return x/255.0

def get_split_data(param):
    print('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed))
    tr_x = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/tr_x.npy')
    tr_y = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/tr_y.npy')
    val_x = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/val_x.npy')
    val_y = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/val_y.npy')
    te_x = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/te_x.npy')
    te_y = np.load('./dataset/'+param.data+'/'+param.run_mode+'/'+param.data+'_rate'+str(param.rate)+'_seed'+str(param.seed)+'/te_y.npy')
    
    tr_y = np.stack([tf.keras.utils.to_categorical(tr_y[:,0], param.num_classes[0]), tf.keras.utils.to_categorical(tr_y[:,1], param.num_classes[1])], axis=1)
    val_y = np.stack([tf.keras.utils.to_categorical(val_y[:,0], param.num_classes[0]), tf.keras.utils.to_categorical(val_y[:,1], param.num_classes[1])], axis=1)
    te_y = np.stack([tf.keras.utils.to_categorical(te_y[:,0], param.num_classes[0]), tf.keras.utils.to_categorical(te_y[:,1], param.num_classes[1])], axis=1)

    return (tr_x, val_x, te_x), (tr_y, val_y, te_y)

def get_data(data, num_classes):
    if data=='mnist':
        dataset = np.load('./dataset/mnist_position.npz', allow_pickle=True)
    if data=='ilab':
        dataset = np.load('./dataset/spandan_ilab_resized.npz', allow_pickle=True)
    if data=='carcgs':
        dataset = np.load('./dataset/relabel_carcgs_resized.npz', allow_pickle=True)
    if data=='daiso':
        dataset = np.load('./dataset/daiso5_resized.npz', allow_pickle=True)

    x, y = dataset['x'], dataset['y']
    y = np.stack([tf.keras.utils.to_categorical(y[:,0], num_classes[0]), 
                  tf.keras.utils.to_categorical(y[:,1], num_classes[1])], axis=1)

    if data != 'mnist':
        x = preprocess(x)
    print(x.shape)
    print(y.shape)
    return x, y


def main():
    print('data loader')

if __name__ == "__main__":
    main()
