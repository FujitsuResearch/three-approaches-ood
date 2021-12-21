import numpy as np
import tensorflow as tf

import sampling
import activation
import tf_logger

def get_random_pair(y):
    tr_pair_idx = sampling.random_pair(y[0])
    val_pair_idx = sampling.random_pair(y[1])
    te_pair_idx = sampling.random_pair(y[2])
    return tr_pair_idx, val_pair_idx, te_pair_idx

def get_pair_tfds(x, y, pair_idx, g_batchsize):
    with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices((x, y, x[pair_idx], y[pair_idx]))
        #ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(len(y)).batch(g_batchsize).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_train_pair_tfds(x, y, pair_idx, g_batchsize):
    with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices((x, y, x[pair_idx], y[pair_idx]))
        #ds = ds.batch(batch_size).prefetch(tf.data.exp3erimental.AUTOTUNE)
        ds = ds.repeat().shuffle(len(y)).batch(g_batchsize).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_pair_distribute_tfds(x, y, pair_idx, g_batchsize, strategy):
    #tr_ds = get_pair_tfds(x[0], y[0], pair_idx[0], g_batchsize)
    tr_ds = get_train_pair_tfds(x[0], y[0], pair_idx[0], g_batchsize)
    tr_ds = strategy.experimental_distribute_dataset(tr_ds)
    val_ds = get_pair_tfds(x[1], y[1], pair_idx[1], g_batchsize)
    val_ds = strategy.experimental_distribute_dataset(val_ds)
    te_ds = get_pair_tfds(x[2], y[2], pair_idx[2], g_batchsize)
    te_ds = strategy.experimental_distribute_dataset(te_ds)
    return tr_ds, val_ds, te_ds

def get_pair_tf_dataset(path, strategy, param, model, x, y, epoch):
    if param.sampling=='random':
        tr_pair_idx, val_pair_idx, te_pair_idx = get_random_pair(y)
    np.save(path+str(epoch)+'tr_pair_idx.npy', tr_pair_idx)
    np.save(path+str(epoch)+'val_pair_idx.npy', val_pair_idx)
    np.save(path+str(epoch)+'te_pair_idx.npy', te_pair_idx)

    tr_ds, val_ds, te_ds = get_pair_distribute_tfds(x, y, (tr_pair_idx, val_pair_idx, te_pair_idx), 
                                                    param.g_batchsize, strategy)
    return tr_ds, val_ds, te_ds

def reset(loss, acc):
    loss[0].reset_states()
    loss[1].reset_states()
    acc.reset_states()
    return


@tf.function
def train_step(norm_type, r_weight, g_batchsize, model, optimizer, 
               criterion, tr_x, tr_y, pair_x, pair_y, tr_loss, tr_acc):
    training = True
    with tf. GradientTape() as tape:
        pred_y, _, out = model(tr_x, training=training)
        _, _, pair_out = model(pair_x, training=False)
        original_norm = tf.norm((out-pair_out), ord=norm_type, axis=-1)
        norm = original_norm * r_weight
        tr_y = tr_y[:,0,:]

        per_example_loss = criterion(tr_y, pred_y)
        loss = per_example_loss + norm
        # replica batch loss(replica batch loss / number of replicas)
        replica_batch_loss = tf.nn.compute_average_loss(loss, global_batch_size=g_batchsize)
        
    gradients = tape.gradient(replica_batch_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    tr_loss[0](per_example_loss)
    tr_loss[1](original_norm)
    tr_acc(tr_y, pred_y)
    return

@tf.function
def test_step(norm_type, r_weight, model, criterion, te_x, te_y, pair_x, pair_y, te_loss, te_acc):
    training=False
    pred_y, _, out = model(te_x, training=training)
    _, _, pair_out = model(pair_x, training=False)
    original_norm = tf.norm((out-pair_out), ord=norm_type, axis=-1)
    norm = original_norm * r_weight
    te_y = te_y[:,0,:]
    loss = criterion(te_y, pred_y)
    te_loss[0](loss)
    te_loss[1](original_norm)
    te_acc(te_y, pred_y)
    return

@tf.function
def distributed_train_step(strategy, norm_type, r_weight, g_batchsize, model, optimizer, 
                           criterion, tr_x, tr_y, pair_x, pair_y, tr_loss, tr_acc):
    #train_step = train_step
    #train_step = tf.function(train_step)
    strategy.run(train_step, args=(norm_type, r_weight, g_batchsize, model, optimizer, 
                                   criterion, tr_x, tr_y, pair_x, pair_y, tr_loss, tr_acc))
    return

@tf.function
def distributed_test_step(strategy, norm_type, r_weight, model, criterion, te_x, te_y, pair_x, pair_y, te_loss, te_acc):
    strategy.run(test_step, args=(norm_type, r_weight, model, criterion, te_x, te_y, pair_x, pair_y, te_loss, te_acc))
    return

def tr_feed_data(ds, strategy, param, model, optimizer, criterion, tr_loss, tr_acc):
    for i, (tr_x, tr_y, pair_x, pair_y) in enumerate(ds):
        distributed_train_step(strategy, param.norm_type, param.r_weight, param.g_batchsize, model, 
                               optimizer, criterion, tr_x, tr_y, pair_x, pair_y, tr_loss, tr_acc)
    return

def tr_feed_repeat_data(ds, strategy, param, model, optimizer, criterion, tr_loss, tr_acc):
    #print('repeat dataset')
    ds = iter(ds)
    # round up
    iterations = -(-param.d_size[0] // param.g_batchsize)
    for i in range(iterations):
        (tr_x, tr_y, pair_x, pair_y) = next(ds)
        distributed_train_step(strategy, param.norm_type, param.r_weight, param.g_batchsize, model, 
                               optimizer, criterion, tr_x, tr_y, pair_x, pair_y, tr_loss, tr_acc)
    return

def te_feed_data(ds, strategy, param, model, criterion, te_loss, te_acc):
    for i, (te_x, te_y, pair_x, pair_y) in enumerate(ds):
        distributed_test_step(strategy, param.norm_type, param.r_weight, model, 
                              criterion, te_x, te_y, pair_x, pair_y, te_loss, te_acc)
    return
