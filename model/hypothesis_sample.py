
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

class Solution:
    def __init__(self) -> object:
        self.W_history = []
        self.cost_history = []
        self.X = []
        self.Y = []

    def hook(self):
        self.create()
        self.checkchart()

    def create(self):
        tf.set_random_seed(777)

        self.X = [1, 2, 3]
        self.Y = [1, 2, 3]

        W = tf.placeholder(tf.float32)
        hypothesis = self.X * W

        cost = tf.reduce_mean(tf.square(hypothesis - self.Y))
        sess = tf.Session()

        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = sess.run(cost, {W: curr_W})
            self.W_history.append(curr_W)
            self.cost_history.append(curr_cost)

    def checkchart(self):
        # 차트로 확인
        plt.plot(self.W_history, self.cost_history)
        plt.show()


if __name__=='__main__':
    Solution().hook()