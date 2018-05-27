import tensorflow as tf 
from module import *
from hyperparams import Hyperparams as hp

def Listener(input_x):
    output = BLSTM(input_x, hp.hidden_units, bidirectional=True)
    output = pBLSTM(input_x, hp.hidden_units)
    output = pBLSTM(input_x, hp.hidden_units)

    return output

def Speller(decoder_input, encoder_state):
    memory = encoder_sstate
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hp.hidden_units)
    vocab_size = len(vocab)
    def step(previous_output, current_input):
        """
        previous_output: [output, previous_weight, state]
        current_input : [decoder_input]
        """
        decoder_current_input = current_input[0]
        previous_weight = previous_output[1]
        previous_state = previous_output[2]

        lstm_output, current_state = lstm_cell(decoder_current_input, previous_state)
        context, current_weight = do_attention(current_output, memory, previous_weight, hp.attention_hidden_units)
        current_output = tf.dense(tf.concat([lstm_output, context], -1), vocab_size)

        return current_output, current_weight, current_state

    decoder_input_scan = tf.transpose(decoder_input, [1,0,2])

    batch_size = tf.shape(decoder_input)[0]
    output_init = tf.zeros([batch_size, vocab_size])
    weight_init = tf.zeros([batch_size, tf.shape(encoder_state)[-1]])
    temp = tf.zeros([tf.shape(decoder_input)[0],hp.hidden_units])
    lstm_state_init = tf.contrib.rnn.LSTMStateTuple(*[temp]*2)

    init = [output_init, weight_init, lstm_state_init]    
    output, attention_weight, _ = tf.scan(decoder_input_scan, initializer=init)
    output = tf.transpose(output, [1,0,2])
    attention_weight = tf.transpose(attention_weight, [1,0,2])

    return output, attention_weight





    