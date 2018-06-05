import tensorflow as tf

def LSTM(input_x, hidden_units, bidirectional=False):
    if bidirectional:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units/2)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units/2)
        encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw_cell,
            cell_bw=lstm_bw_cell,
            inputs = input_x,
            sequence_length = None,
            dtype=tf.float32)
        outputs = tf.concat(encoder_outputs, 2)
    else:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        encoder_outputs, state = tf.nn.dynamic_rnn(
            cell=lstm_fw_cell,
            inputs = input_x,
            sequence_length = None,
            dtype=tf.float32)
        outputs = encoder_outputs

    return outputs

def pBLSTM(input_x, hidden_units):
    static_dim = input_x.get_shape().as_list()
    dynamic_dim = tf.shape(input_x)
    input_x = tf.reshape(input_x, [dynamic_dim[0], tf.cast(dynamic_dim[1]/2, tf.int32), static_dim[-1]*2])

    outputs = LSTM(input_x, hidden_units, bidirectional=True)
    return outputs

def do_attention(state, memory, prev_weight, attention_hidden_units, memory_length=None, reuse=False):
    """
    bahdanau attention, aka, original attention
    state: [batch_size x hidden_units]
    memory: [batch_size x T x hidden_units]
    prev_weight: [batch_size x T]
    """
    state_proj = tf.layers.dense(state, attention_hidden_units, use_bias=True)
    memory_proj = tf.layers.dense(memory, attention_hidden_units, use_bias=None)
    previous_feat = tf.layers.conv1d(inputs=tf.expand_dims(prev_weight,axis=-1), filters=10, kernel_size=50, padding='same')
    previous_feat = tf.layers.dense(previous_feat, attention_hidden_units, use_bias=None)
    temp = tf.expand_dims(state_proj, axis=1) + memory_proj + previous_feat
    temp = tf.tanh(temp)
    score = tf.squeeze(tf.layers.dense(temp, 1, use_bias=None),axis=-1)

    #mask
    if memory_length is not None:
        mask = tf.sequence_mask(memory_length, tf.shape(memory)[1])
        paddings = tf.cast(tf.fill(tf.shape(score), -2**30),tf.float32)
        score = tf.where(mask, score, paddings)

    weight = tf.nn.softmax(score) #[batch x T]
    context_vector = tf.matmul(tf.expand_dims(weight,1),memory)
    context_vector = tf.squeeze(context_vector,axis=1)

    return context_vector, weight

def do_attention_dot(state, memory, prev_weight, attention_hidden_units, memory_length=None, reuse=False):
    """
    dot attention
    state: [batch_size x hidden_units]
    memory: [batch_size x T x hidden_units]
    prev_weight: [batch_size x T]
    """
    state_proj = tf.layers.dense(state, memory.get_shape().as_list()[-1], use_bias=None)
    #memory_proj = tf.layers.dense(memory, attention_hidden_units, use_bias=None)
    memory_proj = memory
    score = tf.reduce_sum(tf.multiply(tf.expand_dims(state_proj, 1), memory_proj), axis=-1)

    #mask
    if memory_length is not None:
        mask = tf.sequence_mask(memory_length, tf.shape(memory)[1])
        paddings = tf.cast(tf.fill(tf.shape(score), -2**30),tf.float32)
        score = tf.where(mask, score, paddings)

    weight = tf.nn.softmax(score) #[batch x T]
    context_vector = tf.matmul(tf.expand_dims(weight,1),memory)
    context_vector = tf.squeeze(context_vector,axis=1)

    return context_vector, weight

def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

#TODO try dot attention



