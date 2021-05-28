import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import numpy as np


class Policy:
    def __init__(self, sess):
        self.sess = sess
        self.state = tf.placeholder(shape=[None, 18], dtype=tf.float32)

        with tf.variable_scope("policy"):
            x = tf.layers.dense(self.state, 32)
            x = tf.nn.tanh(x)
            x = tf.layers.dense(x, 32)
            x = tf.nn.tanh(x)
            out = tf.layers.dense(x, 5)
            self.out = tf.nn.softmax(out)


    def fit(self, state):
        out = self.sess.run(self.out, {self.state: state})

        return out



if __name__ == "__main__":
    with tf.Session() as sess:
        policy = Policy(sess)
        weights_fname = "scripts/data/coop_nav_gcl/policy-150"
        sess.run(tf.global_variables_initializer())
        p = np.random.random(size=(32, 18))
        out = policy.fit(p)
        print(out)
        saver = tf.train.Saver()
        for i, var in enumerate(saver._var_list):
            print('Var {}: {}'.format(i, var))

        for i, var in enumerate(tf.global_variables()):
            print('Var {}: {}'.format(i, var))

        saver.restore(sess, weights_fname)


        """
        Var 0: <tf.Variable 'policy/dense/kernel:0' shape=(18, 32) dtype=float32_ref>
        Var 1: <tf.Variable 'policy/dense/bias:0' shape=(32,) dtype=float32_ref>
        Var 2: <tf.Variable 'policy/dense_1/kernel:0' shape=(32, 32) dtype=float32_ref>
        Var 3: <tf.Variable 'policy/dense_1/bias:0' shape=(32,) dtype=float32_ref>
        Var 4: <tf.Variable 'policy/dense_2/kernel:0' shape=(32, 5) dtype=float32_ref>
        Var 5: <tf.Variable 'policy/dense_2/bias:0' shape=(5,) dtype=float32_ref>
        """




