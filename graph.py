from hyperparams import Hyperparams as hp
import tensorflow as tf
from network import Listener, Speller
from module import *
from data_load import load_vocab, get_batch

class Graph:
    def __init__(self, mode="train"):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set phase
        self.is_training=True if mode=="train" else False

        # Data Feeding
        # x: melspectrogram. (batch, T, n_mels)
        # y: Text. (N, Tx)

        if mode=="train":
            self.y, self.x, _, _, self.num_batch = get_batch()
        else:
            self.x = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
            self.y = tf.placeholder(tf.int32, shape=(None, None))

        # Get encoder/decoder inputs
        with tf.variable_scope('encoder'):
            self.encoder_output = Listener(self.x)
        with tf.variable_scope('decoder'):
            self.decoder_input = tf.concat((tf.ones_like(self.y[:, :1])*self.char2idx['S'], self.y[:, :-1]), -1)
            self.decoder_input = embed(self.decoder_input, len(hp.vocab), hp.hidden_units, zero_pad=True)
            self.logits, self.attention_weight = Speller(self.decoder_input, self.encoder_output, is_training=self.is_training)

        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, self.char2idx['P']))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))

        # Loss
        self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(self.char2idx)))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
        self.loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

        # Training Scheme
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=())
        #self.lr = hp.lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        ## gradient clipping
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.clipped = []
        for grad, var in self.gvs:
            grad = tf.clip_by_norm(grad, 5.)
            self.clipped.append((grad, var))
        self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

        # Summary
        #tf.summary.scalar('{}/guided_attention_loss'.format(mode), self.guided_attn_loss)
        tf.summary.scalar('{}/loss'.format(mode), self.loss)
        tf.summary.scalar('{}/acc'.format(mode), self.acc)
        tf.summary.scalar('{}/lr'.format(mode), self.lr)
        #tf.summary.image("{}/attention".format(mode), tf.expand_dims(self.alignments, -1), max_outputs=1)
        self.merged = tf.summary.merge_all()
