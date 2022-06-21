
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




class Solution():
    def __init__(self) -> object:
       self.model = os.path.join(basedir, 'model')
       self.data = None
    
    def hook(self):
        self.preprocessing()
        self.create_model()
        

    def preprocessing(self):
        self.data = pd.read_csv('./data/price_data.csv', encoding='UTF-8', thousands=',')
        self.data = np.array(self.data, dtype=np.float32)
        #print(price)
        #price.to_csv('./save/price_save.csv')
    

    def create_model(self):
        sess = tf.Session()
        _ = tf.Variable(initial_value = 'fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.model, 'cabbage_model'), global_step=1000)
        

if __name__=='__main__':
    Solution().hook()