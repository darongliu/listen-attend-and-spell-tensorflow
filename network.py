import tensorflow as tf
from module import *
from hyperparams import Hyperparams as hp

def Listener(input_x, pBLSTM_layer=3):
    with tf.variable_scope('lstm'):
        output = LSTM(input_x, hp.hidden_units, bidirectional=True)
    for i in range(pBLSTM_layer):
        with tf.variable_scope('plstm'+str(i)):
            output = pBLSTM(output, hp.hidden_units)

    return output

def Speller(decoder_input, encoder_state, is_training=True):
    with tf.variable_scope('attention_lstm'):
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
            current_input = tf.layers.dropout(current_input, rate=hp.dropout_rate, training=is_training)

            current_output, current_state = lstm_cell(current_input, previous_state)
            if hp.attention_mechanism == 'dot':
                current_context, current_weight = do_attention_dot(current_output, memory, previous_weight, hp.attention_hidden_units)
            else:
                current_context, current_weight = do_attention(current_output, memory, previous_weight, hp.attention_hidden_units) #original attention

            return [current_output, current_context, current_weight, current_state]

        batch_size = tf.shape(decoder_input)[0]
        output_init = tf.zeros([batch_size, hp.hidden_units])
        context_init = tf.zeros([batch_size, hp.hidden_units])
        weight_init = tf.zeros([batch_size, tf.shape(encoder_state)[1]])
        temp = tf.zeros([batch_size,hp.hidden_units])
        lstm_state_init = tf.contrib.rnn.LSTMStateTuple(*[temp]*2)
        init = [output_init, context_init, weight_init, lstm_state_init]

        decoder_input_scan = tf.transpose(decoder_input, [1,0,2])
        output, context, attention_weight, _ = tf.scan(step, [decoder_input_scan], initializer=init)
        output = tf.transpose(output, [1,0,2])
        context = tf.transpose(context, [1,0,2])
        attention_weight = tf.transpose(attention_weight, [1,0,2])

        output = tf.concat([output, context], -1)
        output = tf.layers.dropout(output, rate=hp.dropout_rate, training=is_training)

    with tf.variable_scope('lstm'):
        output = LSTM(output, hp.hidden_units, bidirectional=False)
        output = tf.layers.dropout(output, rate=hp.dropout_rate, training=is_training)

    with tf.variable_scope('output_mlp'):
        output = tf.layers.dense(output, len(hp.vocab))

    return output, attention_weight






