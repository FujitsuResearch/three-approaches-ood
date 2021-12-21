import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import dataloader
import param_config

def get_tfds(x, g_batchsize):
    with tf.device("/cpu:0"):
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.batch(g_batchsize).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_distribute_tfds(x, g_batchsize, strategy):
    ds = get_tfds(x, g_batchsize)
    ds = strategy.experimental_distribute_dataset(ds)
    return ds

@tf.function
def pred_step(model, x):
    training = False
    _, _, acts = model(x, training=training)
    return acts

@tf.function
def distributed_pred_step(strategy, model, x):
    acts = strategy.run(pred_step, args=(model, x))
    return strategy.gather(acts, axis=0)

def actsaccumulate(x_ds, strategy, model):
    for i, batch_x in enumerate(x_ds):
        act = distributed_pred_step(strategy, model, batch_x)
        if i==0:
            acts = act
        else:
            acts = np.append(acts, act, axis=0)
    return acts

def get_activity(x, g_batchsize, strategy, model):
    x_ds = get_distribute_tfds(x, g_batchsize, strategy)
    acts = actsaccumulate(x_ds, strategy, model)
    return acts

def get_ds(data, num_classes, r_batchsize):
    x, y = dataloader.get_data(data, num_classes)
    all_ds = tf.data.Dataset.from_tensor_slices((x, np.argmax(y, axis=-1)))
    all_ds = all_ds.shuffle(len(x)).batch(r_batchsize)
    return all_ds

def get_allacts(path, param):
    ## single gpu only (TODO multi GPUs)
    all_ds = get_ds(param.data, param.num_classes, param.r_batchsize)
    labels = []
    model = tf.keras.models.load_model(path+'omodel')
    allacts = []
    y_true = []
    y_pred = []
    
    for x, y in all_ds:
        pred_y, _, acts = model(x, training=False)
        allacts.extend(acts.numpy())
        y_true.extend(y[:,0])
        y_pred.extend(np.argmax(pred_y, axis=-1))
        labels.extend(y)

    labels = np.array(labels) 
    cmx = confusion_matrix(y_true, y_pred)
    allacts = np.array(allacts)
    np.save(path+'cmx.npy', cmx)
    np.save(path+'allacts.npy', allacts)
    np.save(path+'labels.npy', labels)
    return

def main():
    param = param_config.init_param()

    # set path to trained model
    if param.r_weight == 0.0:
        PATH = './results/vanilla/{0}/{1}/rate{2}_seed{3}_lr{4}_bnm{5}_rw{6}_pi{7}_id{8}_trial{9}/'.format(param.run_mode, param.data, param.rate, param.seed, param.lr, param.batch_momentum, 
                                                                                                                             param.r_weight, param.pairing_interval, param.job_id, param.trial)
    if param.r_weight != 0.0:
        PATH = './results/{0}/{1}/{2}/rate{3}_seed{4}_lr{5}_bnm{6}_rw{7}_pi{8}_id{9}_trial{10}/'.format(param.sampling, param.run_mode, param.data,
                                                                                                                          param.rate, param.seed, param.lr, param.batch_momentum, param.r_weight,
                                                                                                                          param.pairing_interval, param.job_id, param.trial)


    print(PATH)
    get_allacts(PATH, param)                

if __name__ == "__main__":
    main()
