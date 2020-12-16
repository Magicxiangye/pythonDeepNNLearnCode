
#双向循环神经网络的搭建实验
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
#测试常用数据集
from sklearn import datasets

#定义一个循环神经网络的类
class biRnn(object):
    #模型的流程
    def inference(self, x, n_in=None, n_time=None, n_hidden=None, n_out=None):
        #还是先设置权重矩阵和偏移量的初始化
        def weight_variable(shape):
            initial = tf.truncated_normal(shape,stddev=0.01)#截断分布
            return tf.Variable(initial)

        def bias_vaariable(shape):
            initial = tf.zeros(shape,dtype=tf.float32)
            return tf.Variable(initial)

        #先处理一下输入的数据（rnn输入数据是一个张量的list，要把三维的数据处理掉）
        x = tf.transpose(x, [1, 0, 2])#先使用transpose维度的交换（第一和第二位的交换）
        #这样相同时间点t上，所有的图片的数据都在一起
        # 变形为(小批量的大小, 輸入维度)这样的包含时长的形式
        #x最终变为一个张量的list，这样就可以传入rnn
        x = tf.reshape(x, [-1, n_in])

        #定义双向循环的前向层和后向层（BiRNN的隐藏层有两个）
        cell_forward = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)#开启遗忘门
        cell_backward = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)#遗忘门还是要开启的

        #进入rnn网络后的输出
        outputs,_,_ = rnn.stack_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

        #最后的隐层到输出层（再过一个线性的权重加偏置再加激活函数）
        w = weight_variable([n_hidden * 2, n_out])#权重的维度是n_hidden * 2（后面再深入的了解一下）
        b = bias_vaariable([n_out])#有广播机制
        y = tf.nn.softmax(tf.matmul(outputs[-1], w) + b)

        return y


    #误差分析
    def loss(self,y,yPredict):
        #还是用的交叉熵误差函数
        #tf的几个函数回头记一下，学习一下
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(yPredict, 1e-10, 1.0)), reduction_indices=[1]))
        return cross_entropy

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
