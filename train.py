import tf_logger
import train_func

def train(path, strategy, param, model, optimizer, criterion, 
          tr_loss, tr_acc, val_loss, val_acc, te_loss, te_acc, x, y):
    train_summary_writer, valid_summary_writer, test_summary_writer = tf_logger.get_summarywriter(path)
    
    tr_loss_hist, tr_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []
    te_loss_hist, te_acc_hist = [], []

    tr_ds, val_ds, te_ds = train_func.get_pair_tf_dataset(path, strategy, param, model, x, y, 0)

    for epoch in range(param.epoch):
        train_func.reset(tr_loss, tr_acc)
        train_func.reset(val_loss, val_acc)
        train_func.reset(te_loss, te_acc)
        
        if (epoch%param.pairing_interval==0) & (epoch!=0):
            model.save(path+str(epoch)+'omodel')
            tr_ds, val_ds, te_ds = train_func.get_pair_tf_dataset(path, strategy, param, model, x, y, epoch)
        
        print('Lambda', param.r_weight)

        #train_func.tr_feed_data(tr_ds, strategy, param, model, optimizer, criterion, tr_loss, tr_acc)
        train_func.tr_feed_repeat_data(tr_ds, strategy, param, model, optimizer, criterion, tr_loss, tr_acc)
        train_func.te_feed_data(val_ds, strategy, param, model, criterion, val_loss, val_acc)
        train_func.te_feed_data(te_ds, strategy, param, model, criterion, te_loss, te_acc)
        
        tf_logger.write_log(train_summary_writer, valid_summary_writer, test_summary_writer,
                            tr_loss, tr_acc, val_loss, val_acc, te_loss, te_acc, epoch)

        avg_loss = [tr_loss[0].result(), tr_loss[1].result()]
        avg_acc = tr_acc.result()
        avg_valid_loss = [val_loss[0].result(), val_loss[1].result()]
        avg_valid_acc = val_acc.result()
        avg_test_loss = [te_loss[0].result(), te_loss[1].result()]
        avg_test_acc = te_acc.result()
        tr_loss_hist.append(avg_loss)
        tr_acc_hist.append(avg_acc)
        val_loss_hist.append(avg_valid_loss)
        val_acc_hist.append(avg_valid_acc)
        te_loss_hist.append(avg_test_loss)
        te_acc_hist.append(avg_test_acc)
        n = epoch + 1
        template = ("Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}, Test Loss: {}, Test Accuracy: {}")
        print(template.format(n, avg_loss[0]+param.r_weight*avg_loss[1], avg_acc, avg_valid_loss[0]+param.r_weight*avg_valid_loss[1], avg_valid_acc, avg_test_loss[0]+param.r_weight*avg_test_loss[1], avg_test_acc))

    return tr_loss_hist, tr_acc_hist, val_loss_hist, val_acc_hist, te_loss_hist, te_acc_hist
