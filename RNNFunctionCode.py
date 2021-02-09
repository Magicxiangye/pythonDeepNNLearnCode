import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

#未解决的问题（训练的输出中，train_step输出的值为空(但好像就是空,应该没啥问题)，其他参数的输出都正常）
#不知道因为什么

class RNN(object):
    def inference(self,x,n_batch,maxlen=None,n_hidden=None,n_out=None):
        def weight_variable(shape):
            #设置权重矩阵的初始化
            initial = tf.truncated_normal(shape,stddev=0.01)
            return tf.Variable(initial)

        def bias_variable(shape):
            #设置初始的偏移量
            initial = tf.zeros(shape,dtype=tf.float32)
            return tf.Variable(initial)
        #循环网络的初始化
        #传统的循环神经网络没有门阀的控制
        #cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
        #  这个是没有窥视孔功能的LSTM（.LSTMCell()是有窥视孔功能的，参数也是一样用）
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
        initial_state = cell.zero_state(n_batch,tf.float32)

        state = initial_state#隐藏层的初始化状态
        outputs = []#保存过去隐藏层的输出
        with tf.variable_scope('RNN'):
            for t in range(maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:,t, :], state)#三位数组的数据的提取(隐藏层的状态和输入的数据进入RNN网络)
                outputs.append(cell_output)

        output = outputs[-1]#list的最后一项就是隐藏层的输出
        #隐藏层到输出层
        V = weight_variable([n_hidden,n_out])#输出的权重矩阵
        c = bias_variable([n_out])
        #线性的输出
        y = tf.matmul(output,V) + c#线性的激活

        return y

    def loss(self,y,yPredict):
        #使用的平方误差函数
        mse = tf.reduce_mean(tf.square(yPredict - y))
        return mse

    def training(self,loss):
        #使用的是Adam优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
        train_step = optimizer.minimize(loss)
        return train_step

    # 训练的部分
    def fit(self,xTrain,yTrain,maxlen):
        n_in  = len(xTrain[0][0])#1
        n_hidden = 20
        n_out = len(yTrain[0])#1

        #设置占位的参数
        x = tf.placeholder(dtype=tf.float32,shape=[None,maxlen,n_in])#三维的数据进入，来训练（n_in是随着数据的维数变化而变化的）
        y = tf.placeholder(dtype=tf.float32,shape=[None,n_out])#输出维数的要求
        n_batch = tf.placeholder(tf.int32,[])

        #网络的流程
        yPredict = self.inference(x=x,maxlen=maxlen,n_hidden=n_hidden,n_out=n_out,n_batch=n_batch)
        loss = self.loss(y=y,yPredict=yPredict)
        train_step = self.training(loss)

        #开始训练
        epochs = 50 #迭代的次数
        batch_size = 10#小批量的训练

        #初始化会话
        initial  = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initial)

        num_xTrain = int(len(xTrain))
        n_batches = num_xTrain // batch_size#整分多少个批次
        print(n_batches)
        results = []

        for epoch in range(epochs):
            X_,Y_ = shuffle(xTrain,yTrain)

            for i in range(n_batches):
                #数据的起始位置
                start = i * batch_size
                end  = start + batch_size
                print('turn ',i)

                result = sess.run([train_step,loss],feed_dict={x:X_[start:end],y:Y_[start:end],n_batch:batch_size})
                print(result)
                results.append(result)

        return results
            #用验证集进行评估
            #val_loss = loss.eval(session=sess,feed_dict={x:xTrain,y:yTrain,n_batch:xTrain[0]})



if __name__ == '__main__':
    #先生成一下训练的数据
    def sin(x,T=100):
        return np.sin(2.0 *np.pi * x /T)
    def toy_problem(T=100,ampl=0.05):
        num = (2*T) +1
        x = np.arange(0,num)
        noise = ampl * np.random.uniform(low=-1.0,high=1.0,size=len(x))
        return sin(x) +noise


    T = 100
    f = toy_problem(T)

    length_of_squences = 2 * T
    maxlen = 25
    data = []
    target = []
    for i in range(0,length_of_squences - maxlen + 1):
        data.append(f[i:i+maxlen])
        target.append(f[i+maxlen])
    print(data)
    print(target)
#准备数据
    X = np.zeros(dtype=float, shape=(len(data), maxlen, 1))
    Y = np.zeros(dtype=float, shape=(len(data), 1))

    for i ,seq in enumerate(data):
        for t,value in enumerate(seq):
            X[i, t, 0] = value
            Y[i,0] = target[i]

    model = RNN()
    result = model.fit(xTrain=X,yTrain=Y,maxlen=maxlen)
    print(result)





