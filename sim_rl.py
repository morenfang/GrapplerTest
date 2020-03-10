import tensorflow as tf
from tensorflow.contrib.compiler import xla
import numpy as np
import os

device_name = '/device:XLA_CPU:0'  # '/device:XLA_DTU:0'
# '/job:localhost/replica:0/task:0/device:XLA_DTU:0'  # '/device:XLA_DTU:0' '/cpu:0' or '/device:XLA_CPU:0'
data_type = tf.float32  # tf.float16 or tf.float32

# prepare training data
np.random.seed(12)
x_input = np.random.rand(1024, 7170)  # feature vector
x_input.astype(dtype=np.float32)
np.random.seed(15)
image_input = np.random.randint(255, size=(1024, 11, 11, 6))  # image input
y_label = np.zeros([1024, 18])
np.random.seed(16)
index_1024_18 = np.random.randint(0, 18, [1024, ])
for idx, i in enumerate(index_1024_18):
    y_label[idx][i] = 1.0
# obs = tf.Variable(x_input, dtype=data_type)

learning_rate = 0.01
n_steps = 1
n_inputs = 128
n_hiddens = 128
n_classes = 18
training_steps = 1000
batch_sizes = 64
display_steps = 100

# model input size
env_size = 32
hero_size = 128


def linear(input, out_features, bias=True):
    in_features = input.shape[1].value
    bias = tf.truncated_normal([out_features], dtype=data_type)
    weight = tf.truncated_normal([in_features, out_features], dtype=data_type)
    if bias is not None:
        return tf.nn.softmax(tf.matmul(input, weight) + bias)
    else:
        return tf.nn.softmax(tf.matmul(input, weight))


def lstm_train():
    tf.reset_default_graph()

    # tensor placeholder
    with tf.name_scope('inputs'):
        in_env = tf.placeholder(data_type, [None, env_size], name='in_env')
        in_hero = tf.placeholder(data_type, [None, hero_size], name='in_hero')
        in_image = tf.placeholder(data_type, [None, 11, 11, 6], name='in_image')
        # TODO ... other inputs
        out_action = tf.placeholder(data_type, [None, n_classes], name='out_action')

    # weights and biases
    with tf.name_scope('weights'):
        weights = tf.truncated_normal([n_hiddens, n_classes], dtype=data_type, stddev=0.1)
        tf.summary.histogram('output_layer_weights', weights)
        weight_1 = tf.truncated_normal([4, 4, 6, 32], dtype=data_type, stddev=0.1)
        weight_2 = tf.truncated_normal([2, 2, 32, 64], dtype=data_type, stddev=0.1)
        weight_3 = tf.truncated_normal([2, 2, 64, 64], dtype=data_type, stddev=0.1)
    with tf.name_scope('biased'):
        biases = tf.truncated_normal([n_classes], dtype=data_type)
        tf.summary.histogram('output_layer_biased', biases)
        bias_1 = tf.truncated_normal([32], dtype=data_type)
        bias_2 = tf.truncated_normal([64], dtype=data_type)
        bias_3 = tf.truncated_normal([64], dtype=data_type)

    with tf.device(device_name):
        def lstm(_x, _weight, _bias):
            _x = tf.reshape(_x, [-1, n_steps, n_inputs])
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hiddens, state_is_tuple=True, dtype=data_type)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
            _init_state = cell.zero_state(batch_sizes, dtype=data_type)

            outputs, states = tf.nn.dynamic_rnn(cell, _x, initial_state=_init_state, dtype=data_type, time_major=False)
            return tf.nn.softmax(tf.matmul(outputs[:, -1, :], _weight) + _bias)

        out_1 = tf.nn.relu(tf.nn.conv2d(in_image, weight_1, strides=2, padding='VALID') + bias_1)
        out_2 = tf.nn.relu(tf.nn.conv2d(out_1, weight_2, strides=2, padding='VALID') + bias_2)
        out_3 = tf.nn.relu(tf.nn.conv2d(out_2, weight_3, strides=1, padding='VALID') + bias_3)
        in_lstm = tf.concat([tf.reshape(out_3, [batch_sizes, -1]), tf.nn.relu(linear(in_env, 32)), tf.nn.relu(linear(in_hero, 32))], axis=1)

        def model():
            _pred = lstm(in_lstm, weights, biases)
            _cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=out_action))
            _optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(_cost)
            return _pred, _cost, _optimizer

        [op, cost] = xla.compile(model)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir="./logs", graph=sess.graph)
            # Run the initializer
            sess.run(init)
            writer.close()
            for step in range(0, training_steps):
                idx = step % (1024 // 128)
                b_env = x_input[idx * batch_sizes:(idx + 1) * batch_sizes, 0:env_size]
                b_hero = x_input[idx * batch_sizes:(idx + 1) * batch_sizes, env_size:env_size + hero_size]
                b_image = image_input[idx * batch_sizes:(idx + 1) * batch_sizes]
                b_image = b_image.astype(np.float32) / 255.
                b_out = y_label[idx * batch_sizes:(idx + 1) * batch_sizes]
                _, loss = sess.run([op, cost], feed_dict={in_env: b_env, in_hero: b_hero, in_image: b_image, out_action: b_out})
                if step % display_steps == 0:
                    with tf.device('/cpu:0'):
                        correct_prediction = tf.equal(tf.argmax(op, 1), tf.argmax(out_action, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print("Prediction accuracy after training: %s" %
                          sess.run(accuracy,
                              feed_dict={in_env: b_env, in_hero: b_hero, in_image: b_image, out_action: b_out}), 'loss: ', loss)
            print("Optimization Finished!")



def main():
    print(tf.__version__)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    lstm_train()
    '''
    env = tf.placeholder(dtype=data_type, shape=[batch_sizes, env_size], name='env_obs')
    hero = tf.placeholder(dtype=data_type, shape=[batch_sizes, hero_size], name='hero_obs')
    test = tf.placeholder(dtype=data_type, shape=[batch_sizes, 32])
    env = tf.slice(x_input, [0, 0], [2, 3])
    out = linear(env, 4)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ret = sess.run(out)
    print(ret.shape, ret)
    print(sess.run(x_input))
    '''


if __name__ == '__main__':
    main()