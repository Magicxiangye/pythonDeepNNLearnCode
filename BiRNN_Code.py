
#双向循环神经网络的搭建实验
import tensorflow as tf
import numpy as np
#测试常用数据集
from sklearn import datasets

#定义一个循环神经网络的类
class biRnn(object):
    #初始化(参数还没设置别忘了)
    def inference(self):
        #还是先设置权重矩阵和偏移量的初始化
        def weight_variable(shape):
            initial = tf.truncated_normal(shape,stddev=0.01)#截断分布
            return tf.Variable(initial)

        def bias_vaariable(shape):
            initial = tf.zeros(shape,dtype=tf.float32)
            return tf.Variable(initial)
    #误差分析
    def loss(self):
        pass
    #优化器
    def training(self):
        pass
    #训练
    def fit(self):
        pass


if __name__ == '__main__':
    # 获取ML的常用测试数据集MNIST(灰度图的数据集)
    # data_home保存的路径：这里我直接保存在了主路径
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    # MNIST打乱顺序抽取出N数据来进行测试
    n = len(mnist.data)
    N = 1000  # 选取1000个来当神经网络的数据
    # 取出的index的list
    indices = np.random.permutation(range(n))[:N]

    # 准备数据（先做近似min-max的归一化处理（公式在书上））
    X = mnist.data[indices]
    X = X / 255.0
    X = X - X.mean(axis=1).reshape(len(X), 1)
    # 做三维的数据
    X = X.reshape(len(X), 28, 28)  # 转换为时间序列数据
