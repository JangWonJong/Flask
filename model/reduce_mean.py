import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

class Solution:
    def __init__(self) -> object:
        self.num_points = 1000
        self.vectors_set = []
        self.W = None
        self.b = None
        self.x_data = []
        self.y_data = []
        self.train = None

    def hook(self):
        self.create()
        self.train_model()

    def create(self):

        for i in range(self.num_points):
            x1 = np.random.normal(0.0, 0.55)
            y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
            self.vectors_set.append([x1, y1])  # 차원누락

        self.x_data = [v[0] for v in self.vectors_set]
        self.y_data = [v[1] for v in self.vectors_set]

        self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        self.b = tf.Variable(tf.zeros([1])) # zeros 는 배열내부를 0으로 초기화하라
        y = self.W * self.x_data + self.b
        loss = tf.reduce_mean(tf.square(y - self.y_data)) # 경사하강법

        optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 는 학습속도
        self.train = optimizer.minimize(loss)
        return self.train

    def train_model(self):

        init = tf.global_variables_initializer() # 변수초기화. 텐서플로는 반드시 변수초기화 필요
        sess = tf.Session()
        sess.run(init)

        for i in range(8):
            sess.run(self.train)
            print(sess.run(self.W), sess.run(self.b))
            plt.plot(self.x_data, self.y_data, 'ro')
            plt.plot(self.x_data, sess.run(self.W) * self.x_data + sess.run(self.b))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        plt.plot(self.x_data, self.y_data, 'ro')
        plt.plot(self.x_data, sess.run(self.W) * self.x_data + sess.run(self.b))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__=='__main__':
    Solution().hook()