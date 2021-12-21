import datetime
import tensorflow as tf

def get_summarywriter(path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = path + 'logs/gradient_tape/' + current_time + '/train'
    valid_log_dir = path + 'logs/gradient_tape/' + current_time + '/valid'
    test_log_dir = path + 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, valid_summary_writer, test_summary_writer

def write_log(train_summary_writer, valid_summary_writer, test_summary_writer,
              tr_loss, tr_acc, val_loss, val_acc, te_loss, te_acc, epoch):
    with train_summary_writer.as_default():
        tf.summary.scalar('CCE_loss_obj', tr_loss[0].result(), step=epoch)
        tf.summary.scalar('Norm_loss_obj', tr_loss[1].result(), step=epoch)
        tf.summary.scalar('acc_obj', tr_acc.result(), step=epoch)
    with valid_summary_writer.as_default():
        tf.summary.scalar('CCE_loss_obj', val_loss[0].result(), step=epoch)
        tf.summary.scalar('Norm_loss_obj', val_loss[1].result(), step=epoch)
        tf.summary.scalar('acc_obj', val_acc.result(), step=epoch)
    with test_summary_writer.as_default():
        tf.summary.scalar('CCE_loss_obj', te_loss[0].result(), step=epoch)
        tf.summary.scalar('Norm_loss_obj', te_loss[1].result(), step=epoch)
        tf.summary.scalar('acc_obj', te_acc.result(), step=epoch)
    return


def main():
    print('logging')

if __name__ == "__main__":
    main()
