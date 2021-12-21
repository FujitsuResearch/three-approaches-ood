import sys
import os
import random
import time
import datetime
import json

import tensorflow as tf
import numpy as np

import param_config
import dataloader
import activation
import train
from models import separate_cnn
from models import together_resnet_old as together_resnet

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

def main():
    start_time = time.perf_counter()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
    print("start", current_time)
    param = param_config.init_param()
    if param.r_weight == 0.0:
        PATH = './results/vanilla/{0}/{1}/rate{2}_seed{3}_lr{4}_bnm{5}_rw{6}_pi{7}_id{8}_trial{9}/'.format(param.run_mode, param.data, param.rate, param.seed, 
                                                                                                                             param.lr, param.batch_momentum, param.r_weight, 
                                                                                                                             param.pairing_interval, param.job_id, param.trial)
    if param.r_weight != 0.0:
        PATH = './results/{0}/{1}/{2}/rate{3}_seed{4}_lr{5}_bnm{6}_rw{7}_pi{8}_id{9}_trial{10}/'.format(param.sampling, param.run_mode, param.data, 
                                                                                                                          param.rate, param.seed, param.lr, param.batch_momentum,
                                                                                                                          param.r_weight, param.pairing_interval, param.job_id, param.trial)
    os.makedirs(PATH, exist_ok=True)
    print(PATH)
    if os.path.isdir(PATH+'omodel'):
        print('results exist')
        sys.exit()
    # for git log
    #config = param_config.get_dict_param(param)
    #with open(PATH + 'config.json', 'w') as f:
    #    json.dump(config, f, indent=2)

    # set seed
    os.environ["PYTHONHASHSEED"] = str(param.seed)
    random.seed(param.seed)
    np.random.seed(param.seed)
    tf.random.set_seed(param.seed)

    gpu_ids = tf.config.experimental.list_physical_devices('GPU')
    print(gpu_ids)
    if len(gpu_ids) > 0:
        for k in range(len(gpu_ids)):
            tf.config.experimental.set_memory_growth(gpu_ids[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(gpu_ids[k]))
    else:
        print("Not enough GPU hardware devices available")
        
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = param.r_batchsize * strategy.num_replicas_in_sync
    if param.g_batchsize != global_batch_size:
        print('Unexpected number of GPUs')
        #sys.exit()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
    print("start data load", current_time)

    with tf.device("/cpu:0"):
        x, y = dataloader.get_split_data(param)
        param.d_size = [y[0].shape[0], y[1].shape[0], y[2].shape[0]]
        print(param.d_size)
        print(x[0].shape)
        print(y[0].shape)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
    print("end data load", current_time)    

    with strategy.scope():
        o_model = together_resnet.ResNet18(param.num_classes, param.batch_momentum)
        optimizer = tf.keras.optimizers.Adam(learning_rate=param.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                            reduction=tf.keras.losses.Reduction.NONE)
    
        o_tr_l = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
        o_tr_a = tf.keras.metrics.CategoricalAccuracy()
        o_val_l = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
        o_val_a = tf.keras.metrics.CategoricalAccuracy()
        o_te_l = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]
        o_te_a = tf.keras.metrics.CategoricalAccuracy()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
        print("start object", current_time)
        o_tr_loss_hist, o_tr_acc_hist, o_val_loss_hist, o_val_acc_hist, o_te_loss_hist, o_te_acc_hist = train.train(PATH, strategy, param,
                                                                                                                    o_model, optimizer, criterion, 
                                                                                                                    o_tr_l, o_tr_a, o_val_l, 
                                                                                                                    o_val_a, o_te_l, o_te_a,
                                                                                                                    x, y)
        
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
    print("start save", current_time)
    #o_model._set_inputs(tr_ds.__iter__().next()[0])
    o_model.save(PATH+'omodel')
    tr_loss = np.array(o_tr_loss_hist)
    tr_acc = np.array(o_tr_acc_hist)
    val_loss = np.array(o_val_loss_hist)
    val_acc = np.array(o_val_acc_hist)
    te_loss = np.array(o_te_loss_hist)
    te_acc = np.array(o_te_acc_hist)

    np.save(PATH+'tr_loss.npy', tr_loss)
    np.save(PATH+'tr_acc.npy', tr_acc)
    np.save(PATH+'val_loss.npy', val_loss)
    np.save(PATH+'val_acc.npy', val_acc)
    np.save(PATH+'te_loss.npy', te_loss)
    np.save(PATH+'te_acc.npy', te_acc)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
    print("end save and all", current_time)
    end_time = time.perf_counter()
    print("elapsed_time:{} [sec]".format(end_time-start_time))
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
    print("start get activation", current_time)
    activation.get_allacts(PATH, param)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")   
    print("end save and all", current_time)
    all_end_time = time.perf_counter()
    print("elapsed_time(get all activation):{} [sec]".format(all_end_time-end_time))


if __name__ == '__main__':
    main()
    
