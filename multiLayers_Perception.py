import tensorflow as tf
import numpy as np
#初始设置(训练数据集)
learnRate = 0.1
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])
#直接设置神经网络
x = tf.placeholder(dtype=tf.float32,shape=[None,2])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])
#第一个隐藏层
w1 = tf.Variable(tf.truncated_normal([2,2]))
print(w1)
b1 = tf.Variable(tf.zeros([2]))
a1 = tf.nn.sigmoid(tf.matmul(x,w1) + b1)
#进入第二个隐藏层(最后输出值是一维的)
w2 = tf.Variable(tf.truncated_normal([2,1]))
b2 = tf.Variable(tf.zeros([1]))
yPredict = tf.nn.sigmoid(tf.matmul(a1,w2) + b2)
#误差函数(还是交叉熵误差函数)
cross_entory = -tf.reduce_sum(y*tf.log(yPredict)+(1-y)*tf.log(1-yPredict))
#优化器
optimizer = tf.train.GradientDescentOptimizer(learnRate)
train = optimizer.minimize(cross_entory)
#验证准确性
correct_prediction = tf.equal(tf.to_float(tf.greater(yPredict,0.5)),y)#tf.greater()返回yPredict>0.5的真值
#会话初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练过程
for epoch in range(4000):
    result  = sess.run([cross_entory,train,w1,b1],feed_dict={x: X, y: Y})
    if epoch % 100 ==0:
        print(result)
#测试一下
classified = correct_prediction.eval(session=sess,feed_dict={x: X, y: Y})
print('it is classified result')
print(classified)
prob = yPredict.eval(session=sess,feed_dict={x: X})
print('it is prob result')
print(prob)