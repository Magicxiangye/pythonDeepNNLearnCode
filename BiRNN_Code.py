
#双向循环神经网络的搭建实验
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
#测试常用数据集
from sklearn import datasets
#应用两个工具
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
        #同时将三维的数据降维，变为[28 * len(x), 28 ]
        x = tf.reshape(x, [-1, n_in])
        # 变形为(小批量的大小, 輸入维度)这样的包含时长的形式()这里没有分批次就是全数据的进入
        # x最终变为一个张量的list，这样就可以传入rnn
        #切割的函数（张量， 分为多少份， axis）
        x = tf.split(x, n_time, 0)

        #定义双向循环的前向层和后向层（BiRNN的隐藏层有两个）
        cell_forward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)#开启遗忘门
        cell_backward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)#遗忘门还是要开启的

        #进入rnn网络后的输出
        outputs,_,_ = rnn.static_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

        #最后的隐层到输出层（再过一个线性的权重加偏置再加激活函数）
        w = weight_variable([n_hidden * 2, n_out])#权重的维度是n_hidden * 2（后面再深入的了解一下）
        b = bias_vaariable([n_out])#有广播机制
        y = tf.nn.softmax(tf.matmul(outputs[-1], w) + b)

        return y


    #误差分析
    def loss(self,y,yPredict):
        #还是用的交叉熵误差函数
        #tf的几个常用的函数
        #tf.clip_by_value()是使张量的值限定在一个值范围里，可以取等
        #tf.reduce_sum（）的reduction_indices参数，是
        cross_entropy = tf.reduce_sum(-tf.reduce_sum(y * tf.log(tf.clip_by_value(yPredict, 1e-10, 1.0)), reduction_indices=[1]))
        return cross_entropy

    #优化器
    def training(self,loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        train = optimizer.minimize(loss)
        return train

    #精确度函数的使用
    def accuracy(self,y,yPredict):
        correct_prediction  = tf.equal(tf.argmax(y,1), tf.argmax(yPredict,1))#每行最大值的索引是否一样(压缩的是第二维)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#张量的类型的转换，最后求总的平均训练的精确度
        return accuracy

    #训练
    #直接主进程写得了，麻烦
    def fit(self):
        pass


if __name__ == '__main__':
    # 获取ML的常用测试数据集MNIST(灰度图的数据集)
    # data_home保存的路径：这里我直接保存在了主路径
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    # MNIST打乱顺序抽取出N数据来进行测试
    n = len(mnist.data)
    N = 3000  # 选取部分MNIST数据进行实验
    N_train = 2000
    N_validation = 400# 选取1000个来当神经网络的数据
    # 取出的index的list
    indices = np.random.permutation(range(n))[:N]

    # 准备数据（先做近似min-max的归一化处理（公式在书上））
    X = mnist.data[indices]
    X = X / 255.0
    X = X - X.mean(axis=1).reshape(len(X), 1)
    # 做三维的数据
    X = X.reshape(len(X), 28, 28)  # 转换为时间序列数据
    #真实集
    Y = mnist.target[indices]
    Y = np.eye(10)[Y.astype(int)]  # 转换为1-of-K形式

    #划分一下训练和测试的数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

    #直接开始写训练的代码
    #定义一下模型中的参数
    n_in = 28
    n_time = 28
    n_hidden = 128
    n_out = 10 #判断是0-9的十个图片数

    #占位的还是来两个
    x = tf.placeholder(shape=[None, n_time, n_in],dtype=tf.float32)
    y = tf.placeholder(shape=[None, n_out], dtype=tf.float32)

    #使用模型的流程
    model = biRnn()
    yPredict = model.inference(x=x, n_out=n_out, n_in=n_in, n_hidden=n_hidden,n_time=n_time)
    loss = model.loss(y=y, yPredict=yPredict)
    train = model.training(loss=loss)
    #精确度
    accuracy = model.accuracy(y=y, yPredict=yPredict)
    #保存所有迭代后的数据，最后做可视化
    history = {
        'val_loss': [],
        'val_acc': []
    }

    #实例化流程后开始训练
    #训练的设置
    epochs = 50
    batch_size = 20

    #初始化会话
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #小批量的训练
    n_batch = N_train // batch_size

    #开始迭代训练
    for epoch in range(epochs):
        #打乱数据
        X_,Y_ = shuffle(X_train, Y_train)

        #开始批次训练
        for i in range(n_batch):
            start = i * batch_size
            end = start + batch_size
            #训练
            sess.run(train, feed_dict={x:X_train[start:end], y:Y_train[start:end]})

        #每轮迭代好后进dev set进行测试
        #误差
        val_loss = loss.eval(session=sess, feed_dict={x:X_validation, y:Y_validation})
        #精确度
        val_accuracy = accuracy.eval(session=sess, feed_dict={x:X_validation, y:Y_validation})

        #加入数组(每次的迭代)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        #在监视器上先看一下
        print('epoch:', epoch, 'val_loss:', val_loss, 'val_accuracy:', val_accuracy)
        #早停法没用，后期可以加上

    '''
       训练进度情况可视化
       '''
    loss = history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss,
             label='loss', color='black')
    plt.xlabel('epochs')
    plt.show()

    '''
    评估预测精度
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_test,
        y: Y_test
    })
    print('accuracy: ', accuracy_rate)



