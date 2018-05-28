import tensorflow as tf 
from module import *
from hyperparams import Hyperparams as hp

def Listener(input_x, pBLSTM_layer=3):
    output = LSTM(input_x, hp.hidden_units, bidirectional=True)
    for _ in range(pBLSTM_layer):
        output = pBLSTM(output, hp.hidden_units)

    return output

def Speller(decoder_input, encoder_state):
    with tf.variable_scope('attention lstm'):
        memory = encoder_state
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hp.hidden_units)
        def step(previous_output, current_input):
            """
            previous_output: [output, previous_context, previous_weight, state]
            current_input : [decoder_input]
            """
            previous_context = previous_output[1]
            previous_weight = previous_output[2]
            previous_state = previous_output[3]

            decoder_current_input = current_input[0]
            current_input = tf.concat([decoder_current_input, previous_context], -1)

            current_output, current_state = lstm_cell(current_input, previous_state)
            current_context, current_weight = do_attention(current_output, memory, previous_weight, hp.attention_hidden_units)

            return current_output, current_context, current_weight, current_state

        batch_size = tf.shape(decoder_input)[0]
        output_init = tf.zeros([batch_size, hp.hidden_units])
        context_init = tf.zeros([batch_size, hp.hidden_units])
        weight_init = tf.zeros([batch_size, tf.shape(encoder_state)[1]])
        temp = tf.zeros([tf.shape(decoder_input)[0],hp.hidden_units])
        lstm_state_init = tf.contrib.rnn.LSTMStateTuple(*[temp]*2)
        init = [output_init, context_init, weight_init, lstm_state_init]    

        decoder_input_scan = tf.transpose(decoder_input, [1,0,2])
        output, context, attention_weight, _ = tf.scan(decoder_input_scan, initializer=init)
        output = tf.transpose(output, [1,0,2])
        context = tf.transpose(context, [1,0,2])
        attention_weight = tf.transpose(attention_weight, [1,0,2])

        output = tf.concat([output, context], -1)

    with tf.variable_scope('lstm'):
        output = LSTM(output, hp.hidden_units, bidirectional=False)

    with tf.variable_scope('output_mlp'):
        output = tf.dense(output, vocab_size)

    return output, attention_weight





    