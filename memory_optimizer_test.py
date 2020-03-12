import tensorflow as tf

# device = "/device:XLA_CPU:0"
device = "/device:XLA_DTU:0"


def memory_optimizer_test():
    with tf.device(device):
        a = tf.random_normal([1000, 1280, 12], dtype=tf.float32, name='morenfang/a')
        b = tf.random_normal([1000, 1280, 12], dtype=tf.float32, name='b')
        c = tf.random_normal([1000, 1280, 12], dtype=tf.float32, name='c')
        # a = tf.zeros([1000, 1280, 12], dtype=tf.float32, name="morenfang/a")
        # b = tf.zeros([1000, 1280, 12], dtype=tf.float32, name="b")
        # c = tf.zeros([1000, 1280, 12], dtype=tf.float32, name="c")
        d = tf.add_n([a, b, c], name='d')
        e = tf.sqrt(d, name="e")

        sess = tf.Session()
        writer = tf.summary.FileWriter(logdir="./logs", graph=sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(e)
        print(out)
        writer.close()


def main():
    memory_optimizer_test()


if __name__ == "__main__":
    main()
