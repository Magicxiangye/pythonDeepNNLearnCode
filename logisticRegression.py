import tensorflow as tf
import numpy as np

learnRate = 0.1
#实现的是逻辑门或门的逻辑回归
#设置下训练的数据
x1 = np.array([[0,0],[0,1],[1,0],[1,1]])
y1 = np.array([[0],[1],[1],[1]])

#logisticRegression的参数
w = tf.Variable(tf.zeros([2,1]))#向量
b = tf.Variable(tf.zeros([1]))#偏移量

#自己定义激活函数
def sigmoid(x):
    result = 1 / (1 + tf.exp(-x))
    return result
def yResult(x):
    z1 = np.dot(w,x) + b
    return z1
#训练的流程
x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])
#前向传播
yPredict = tf.nn.sigmoid(tf.matmul(x,w) + b)
#损失函数的使用（交叉熵误差函数）
cross_entropy = -tf.reduce_sum((y * tf.log(yPredict) + (1-y)*tf.log(1-yPredict)))
optimizer = tf.train.GradientDescentOptimizer(learnRate)#参数更新的方法
train = optimizer.minimize(cross_entropy)
#预测结果的检测
correct_prediction = tf.equal(tf.to_float(tf.greater(yPredict,0.5)),y)#yPredict与0.5比大小结果看与y是否相等
#初始化训练的session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练
for epoch in range(2000):
    # for i in range(x1.shape[0]):
    #     results = sess.run([train, x, y, yPredict, cross_entropy], feed_dict={x: x1[i], y: y1[i]})
    #     print(results)
    results = sess.run([train, x, y, yPredict, cross_entropy], feed_dict={x: x1, y: y1})
    print(results)
#验证训练结果
classified = correct_prediction.eval(session=sess,feed_dict={x:x1,y:y1})
print(classified)
#测试
prob = yPredict.eval(session=sess,feed_dict={x:x1})
print('prob:',prob)
print('w:',sess.run(w))
print("b:",sess.run(b))