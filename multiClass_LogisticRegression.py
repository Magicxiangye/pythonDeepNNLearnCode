import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle#sklearn的打乱顺序功能

#数据准备
M = 2 #输入数据的维度
K = 3 #分类数
n = 100 #每个分类数的数据
N = n * K #训练的总数据数
learnRate = 0.1

#随机生成训练的样本
x1 = np.random.randn(n,M) + np.array([0,10])#cluster1
x2 = np.random.randn(n,M) + np.array([5,5])#cluster2
x3 = np.random.randn(n,M) + np.array([10,0])#cluster3
#训练数据的真实值
y1 = np.array([[1,0,0] for i in range(n)])
y2 = np.array([[0,1,0] for num in range(n)])
y3 = np.array([[0,0,1] for j in range(n)])

#组合数据
X = np.concatenate((x1,x2,x3),axis=0)
Y = np.concatenate((y1,y2,y3),axis=0)

#定义神经网络
#输入数据
x = tf.placeholder(dtype=tf.float32,shape=[None,M])#M个属性
y = tf.placeholder(dtype=tf.float32,shape=[None,K])#分为K类
#训练参数
w = tf.Variable(tf.zeros([M,K]))
b = tf.Variable(tf.zeros([K]))
#前向传播
z1 = tf.matmul(x,w) + b
#softmax函数
yPredict = tf.nn.softmax(z1)
#误差函数还是交叉熵损失函数(用了mini—batch的gradientDescent)
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yPredict),reduction_indices=[1]))#reduction_indices是要加和的方向
optimizer = tf.train.GradientDescentOptimizer(learnRate)
train = optimizer.minimize(cross_entroy)
#检验预测的准确值
correct_prediction  = tf.equal(tf.argmax(yPredict,1),tf.argmax(y,1))#每行最大值的索引是否一样
#初始化会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练流程
batch_size = 50 #batch的size
n_batchs = N // batch_size#分为了几个batch(整除)

for epoch in range(20):
    X_,Y_ = shuffle(X,Y)#每次迭代打乱顺序

    for i in range(n_batchs):
        #定义每个mini_batch的启示位置
        start = i * batch_size
        end = start + batch_size
        #trainProcess
        result = sess.run([train,yPredict,w,b],feed_dict={x:X_[start:end],y:Y_[start:end]})
        print(result)

#测试结果
xPre,yPre = shuffle(X,Y)
#是否判断正确的检测
classified = correct_prediction.eval(session=sess,feed_dict={x:xPre[0:10],y:yPre[0:10]})
print("classified results")
print(classified)
#预测的值
prob = yPredict.eval(session=sess,feed_dict={x:xPre[0:10]})
print("prob num is")
print(prob)



