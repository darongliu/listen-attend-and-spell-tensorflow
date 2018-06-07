'''
modified from:
https://www.github.com/kyubyong/tacotron
'''

import os
import sys
import numpy as np
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm
from utils import *
from graph import Graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("keep_train", "False", "keep training from existed model or not")

if __name__ == '__main__':
    keep_train = FLAGS.keep_train
    g = Graph(); print("Training Graph loaded")
    if not os.path.isdir(hp.logdir):
        os.makedirs(hp.logdir)
    logfile = open(os.path.join(hp.logdir,hp.logfile), "a")
    saver = tf.train.Saver(max_to_keep=10)
    init = tf.global_variables_initializer()
    #sv = tf.train.Supervisor(logdir=hp.logdir, save_summaries_secs=60, save_model_secs=0)
    with tf.Session() as sess:
        #while 1:
        writer = tf.summary.FileWriter(hp.logdir, graph = sess.graph)

        if keep_train == "True":
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Continue training from existed latest model...")
        else:
            sess.run(init)
            print("Initial new training...")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        lr = hp.lr
        previous_total_loss = np.inf
        for epoch in range(1, hp.num_epochs + 1):
            total_loss = 0.0
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                #_, gs = sess.run([g.train_op, g.global_step])
                _, gs, l = sess.run([g.train_op, g.global_step, g.loss], feed_dict={g.lr:lr})

                total_loss += l

                # Write checkpoint files
                if gs % 1000 == 0:
                    #sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))
                    # plot the first alignment for logging
                    al = sess.run(g.attention_weight)
                    plot_alignment(al[0], gs)

            if total_loss > previous_total_loss:
                print('decay learning rate by', hp.lr_decay)
                lr = lr * hp.lr_decay
            previous_total_loss = total_loss

            print("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + "\n")
            sys.stdout.flush()
            logfile.write("Epoch " + str(epoch) + " average loss:  " + str(total_loss/float(g.num_batch)) + "\n")

            # Write checkpoint files
            if epoch % 10 == 0:
                #sv.saver.save(sess, hp.logdir + '/model_gs_{}k'.format(gs//1000))
                saver.save(sess, hp.logdir + '/model_epoch_{}.ckpt'.format(epoch))
                result = sess.run(g.merged)
                writer.add_summary(result, epoch)

        coord.request_stop()
        coord.join(threads)

    print("Done")

# add dropout
# use diff attention


