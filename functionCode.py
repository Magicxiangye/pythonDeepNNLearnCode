import tensorflow as tf
from sklearn.utils import shuffle

#对tensorflow的模型进行类的封装
class DNN(object):
    #初始化模型(接收各个层的维度设定)
    def __init__(self,n_in,n_hiddens,n_out):
        self.n_in = n_in
        self.n_hiddens = n_hiddens#(传入的是一个list，包含着多个隐藏层的维度)
        self.n_out = n_out
        self.weights = []#weight  shape's list
        self.biases = []#bias shape's list
        #为测试函数保存的值先要预设一下
        self._x = None
        self._y = None
        self._yPredict = None
        self._keep_prob = None
        self._sess = None
        self._history = {'accuracy': [],'loss': []}
    #定义的设置权重矩阵的方法（传入的是对应层的shape）
    def weight_Variable(self,shape):
        initial = tf.truncated_normal(shape=shape,stddev=0.01)
        return tf.Variable(initial)
    #定义偏移量的方法
    def bias_variable(self,shape):
        bias_initial = tf.zeros(shape)
        return tf.Variable(bias_initial)
    #定义模型(根据后期的情况，要添加层数的话还得修改代码)
    def inference(self,x,keep_prob=None):
        output = None
        #这里定义的是多层的神经网络（取决与hidden的list中定义的有多少个）
        for i,n_hidden in enumerate(self.n_hiddens):
            #用枚举来返回索引和值
            if i == 0:
                #输入层到第一层的隐藏层
                input = x
                input_dim = self.n_in#即表示的是每条输入数据的参数的维度
            else:
                input  = output#下一层的输入是上一层的输出
                input_dim = self.n_hiddens[i - 1]#i-1是进入上一个隐藏层的维度的索引值
            self.weights.append(self.weight_Variable([input_dim,n_hidden]))  #隐藏层shape的定义[上层的维度，本层的维度]
            self.biases.append(self.bias_variable([n_hidden]))#bias的维度一样就行，有广播机制
            #隐层中使用的激活函数
            z1 = tf.nn.relu(tf.matmul(input,self.weights[-1]) + self.biases[-1])#list是在最后插入的值，所以用倒叙索引
            if keep_prob:
                #是否要使用随机失活的功能
                output = tf.nn.dropout(z1,keep_prob=keep_prob)
            else:
                output = z1
        #出循环后到最后一层（隐层到输出层）
        self.weights.append(self.weight_Variable([self.n_hiddens[-1],self.n_out]))#最后一下的权重矩阵
        self.biases.append(self.bias_variable([self.n_out]))
        yPredict = tf.nn.softmax(tf.matmul(output,self.weights[-1]) + self.biases[-1])
        return yPredict
    #定义误差函数
    def loss(self,y,yPredict):
        cross_entroy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(yPredict,1e-10,1.0)), reduction_indices=[1]))  # reduction_indices是要加和的方向
        return cross_entroy
    #定义训练模型（优化器之类的）
    def training(self,loss):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)
        return train
    #模型精确度的方法
    def accuracy(self,y,yPredict):
        correct_prediction = tf.equal(tf.argmax(yPredict,1),tf.argmax(y,1))#yPredict每行的最大值与真实的比较，看看是不是同一个索引
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#平均的精确度
        return accuracy
    #训练的函数
    def fit(self,X_train,Y_train,epochs=100,batch_size=100,p_keep=0.5,verbose=1):
        #定义传入模型的参数
        x = tf.placeholder(dtype=tf.float32,shape=[None,self.n_in])
        y = tf.placeholder(dtype=tf.float32,shape=[None,self.n_out])
        keep_prob = tf.placeholder(dtype=tf.float32)
        #把用于测试函数的值保存到类的属性中
        self._x = x
        self._y = y
        self._keep_prob = keep_prob
        #训练的流程
        yPredict = self.inference(x,keep_prob)
        loss = self.loss(y=y,yPredict=yPredict)
        train = self.training(loss)
        accracy = self.accuracy(y=y,yPredict=yPredict)#精确度
        #初始化会话
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        #把用于预测的值都先赋值给函数的属性
        self._sess = sess

        #mini-batch训练
        N_train = len(X_train)
        N_batches = N_train // batch_size#整除分为N_batches个批次

        #开始训练
        for epoch in range(epochs):
            X_,Y_ = shuffle(X_train,Y_train)
            #分批次的训练
            for batch in range(N_batches):
                start = batch * batch_size
                end = start + batch_size
                sess.run(train,feed_dict={x:X_[start:end],y:Y_[start:end],keep_prob:p_keep})
            # 记录每epoch的loss
            train_loss = loss.eval(session=sess,feed_dict={x:X_train,y:Y_train,keep_prob:1.0})#要用.eavl的话要先实例化一下函数才行
            train_accracy = accracy.eval(session=sess,feed_dict={x:X_train,y:Y_train,keep_prob:1.0})
            #将结果计入下来
            self._history['loss'].append(train_loss)
            self._history['accracy'].append(train_accracy)

            #可视化一下
            if verbose:
                print('epoch:',epoch,'loss:',train_loss,'accracy:',train_accracy)

        #训练的返回值
        return self._history

     #测试的方法
    def evaluate(self,X_test,Y_test):
        accuracy = self.accuracy(self._yPredict,self._y)
        return accuracy.eval(session=self._sess,feed_dict={self._x:X_test,self._y:Y_test,self._keep_prob:1.0})#小不同点


# 使用方法
# model = DNN(n_in=784,n_hiddens=[100,100,100],n_out=10)
# model.fit(X_train=x,Y_train=y,epochs=50,batch_size=100,p_keep=0.5)
# accruacy = model.evaluate(X_test=x,Y_test=y)