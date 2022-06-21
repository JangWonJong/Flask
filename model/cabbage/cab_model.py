
import os
import sys
from icecream import ic
from torch import dtype
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import torch
import torch.optim as optim


class Cabbage():
    def __init__(self) -> object:
       self.model = os.path.join(basedir, 'model')
       self.data = None
       self.data_test = None
       self.avgTemp = None
       self.minTemp = None
       self.maxTemp = None
       self.rainFall = None
       


    def hook(self):
        #self.cabbage()
        self.preprocessing()
        self.create_model()

    '''def cabbage(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.import_meta_graph(self.model + '/cabbage_model-1000.meta')
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name('w1:0')
            w2 = graph.get_tensor_by_name('w2:0')
            w3 = graph.get_tensor_by_name('w3:0')
            w4 = graph.get_tensor_by_name('w4:0')
            feed_dict = {w1: float(self.avgTemp), w2: float(self.minTemp),  w3: float(self.maxTemp), w4: float(self.rainFall)}
            cb_to_restore = graph.get_tensor_by_name('cb_:0')
            result = sess.run(cb_to_restore, feed_dict)
            print(f'최종결과: {result}')
        return result'''


    def preprocessing(self):
        self.data = pd.read_csv('./data/price_data.csv', encoding='UTF-8', thousands=',')
        self.data = np.array(self.data, dtype=np.float32)
        x_data = self.data[:, 1:-1]
        y_data = self.data[:, [-1]]
        print(x_data)
        print(y_data)
        #print(price)
        #price.to_csv('./save/price_save.csv')
    

    def create_model(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        w = tf.Variable(tf.random_normal([4,1]), name = "weight")
        b = tf.Variable(tf.random_normal([1]), name = "bias")
        hypothesis = tf.matmul(X, w) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())

        '''for step in range(10000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y:y_data})
            if step % 500 == 0:
                print("#", step, "손실비용:", cost_)
                print("-배추가격", hypo_[0])'''

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage_model'), global_step=1000)
        

if __name__=='__main__':
    Cabbage().hook()