
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

class Cabbage():
    def __init__(self) -> object:
       self.data = None
       self.data_test = None
       self.x_data =None
       self.y_data =None

    def preprocessing(self):
        self.data = pd.read_csv('./data/price_data.csv', encoding='UTF-8', thousands=',')
        self.data = np.array(self.data, dtype=np.float32)
        self.x_data = self.data[:, 1:-1]
        self.y_data = self.data[:, [-1]]

    def create_model(self):
        #텐서모델 초기화(모델템플릿 생성)
        self.model = os.path.join(basedir, 'model')
        #확률변수 데이터
        self.preprocessing() 
        #선형식(가설)제작 y= Wx+b
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        w = tf.Variable(tf.random_normal([4,1]), name = "weight")
        b = tf.Variable(tf.random_normal([1]), name = "bias")
        hypothesis = tf.matmul(X, w) + b
        #손실함수
        cost = tf.reduce_mean(tf.square(hypothesis) - Y)
        #최적화 알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00005)
        train = optimizer.minimize(cost)
       
        #세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #트레이닝
        for step in range(10000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print("#", step, "손실비용:", cost_)
                print("-배추가격", hypo_[0])
        #모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage'), global_step = 1000)
        print('저장완료')

    def load_model(self, avgTemp, minTemp, maxTemp, rainFall):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        w = tf.Variable(tf.random_normal([4,1]), name = "weight")
        b = tf.Variable(tf.random_normal([1]), name = "bias")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '')
            data = [[avgTemp, minTemp, maxTemp, rainFall]]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X,w) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])

if __name__=='__main__':
    Cabbage().create_model()