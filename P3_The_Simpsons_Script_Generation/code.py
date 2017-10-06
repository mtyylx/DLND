import helper
import numpy as np
import tensorflow as tf
import problem_unittests as tests


data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab = set(text)
    word2int = {word: idx for idx, word in enumerate(vocab)}
    int2word = {idx: word for idx, word in enumerate(vocab)}

    return word2int, int2word


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punc_dict = {'.' :  '||Period||',
                 ',' :  '||Comma||',
                 '"' :  '||Quotation||',
                 ';' :  '||Semicolon||',
                 '!' :  '||Exclamation||',
                 '?' :  '||Question||',
                 '(' :  '||Left||',
                 ')' :  '||Right||',
                 '--' : '||Dash||',
                 '\n' : '||Return||'}
    return punc_dict


helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
    lr = tf.placeholder(dtype=tf.float32, name='lr')
    return inputs, targets, lr


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    # 只用一个 LSTM Cell 模型，看看效果
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * 3)

    # 用指定batch_size初始化LSTM Cell，并重命名
    init = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    init = tf.identity(init, name='initial_state')
    return cell, init


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embed_init = tf.random_uniform(shape=(vocab_size, embed_dim), minval=-1, maxval=1)
    embedding = tf.Variable(initial_value=embed_init)
    return tf.nn.embedding_lookup(params=embedding, ids=input_data)
    # return tf.contrib.layers.embed_sequence(input_data,
    #                                         vocab_size=vocab_size,
    #                                         embed_dim=embed_dim)


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, finalState = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, dtype=tf.float32)
    finalState = tf.identity(finalState, name='final_state')
    return outputs, finalState


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embed = get_embed(input_data, vocab_size, rnn_size)
    outputs, finalState = build_rnn(cell, embed)
    out = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=vocab_size, activation_fn=None)
    return out, finalState


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function

    n_batches = len(int_text) // (batch_size * seq_length)
    lens = n_batches * batch_size * seq_length
    inputs_X = np.array(int_text[:lens]).reshape((batch_size, -1))
    inputs_Y = np.roll(int_text, -1)[:lens].reshape((batch_size, -1))
    res = np.zeros((n_batches, 2, batch_size, seq_length), dtype=np.int32)

    # 每个batch内分别存储输入和期望输出值对象
    for i in range(n_batches):
        res[i, 0] = inputs_X[:, i * seq_length: (i + 1) * seq_length]
        res[i, 1] = inputs_Y[:, i * seq_length: (i + 1) * seq_length]
    return res

def get_batches2(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text) // (batch_size * seq_length)
    result = []
    for i in range(n_batches):
        inputs = []
        targets = []
        for j in range(batch_size):
            idx = i * seq_length + j * seq_length
            inputs.append(int_text[idx : idx + seq_length])
            targets.append(int_text[idx + 1:idx + seq_length + 1])
        result.append([inputs, targets])
    return np.array(result)

def get_batches3(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    int_text = int_text[16:]
    n_batches = (len(int_text)-1)//(batch_size * seq_length)
    int_text = int_text[:n_batches * batch_size * seq_length + 1]
    int_text_sequences = [int_text[i*seq_length:i*seq_length+seq_length] for i in range(0, n_batches * batch_size)]
    int_text = int_text[1:]
    int_text_targets = [int_text[i*seq_length:i*seq_length+seq_length] for i in range(0, n_batches * batch_size)]
    output = []
    for batch in range(n_batches):
        inputs = []
        targets = []
        for size in range(batch_size):
            inputs.append(int_text_sequences[size * n_batches + batch])
            targets.append(int_text_targets[size * n_batches + batch])
        output.append([inputs, targets])
    return np.array(output)

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 119
# RNN Size
rnn_size = 256
# Embedded Dimensions
embed_dims = 300
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = get_batches3(int_text, batch_size, seq_length).shape[0]

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'


from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dims)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


batches = get_batches3(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')