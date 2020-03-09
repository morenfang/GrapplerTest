import tensorflow as tf

device = "/device:cpu:0"
# device = "/device:XLA_DTU:0"


def memory_optimizer_test():
    a = tf.random_normal([128, 128, 12], dtype=tf.float32, name='a')
    b = tf.random_normal([128, 128, 12], dtype=tf.float32, name='b')
    c = tf.random_normal([128, 128, 12], dtype=tf.float32, name='c')
    d = tf.add_n([a, b, c], name='d')

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    out = sess.run(d)
    print(out)


def main():
    memory_optimizer_test()


if __name__ == "__main__":
    main()
