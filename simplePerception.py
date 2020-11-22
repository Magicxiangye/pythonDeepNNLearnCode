import numpy as np

#产生随机数据集
rng = np.random.RandomState(123)

d = 2 #每个数据的维度
N = 10 #每种数据的数量
mean = 5 #可以激活神经元的那一组

x1 = rng.randn(N,d) + np.array([0,0])
x2 = rng.randn(N,d) + np.array([5,5])
#合成一个训练集
x = np.concatenate((x1,x2),axis=0)
#初始化权重和偏置量
w = np.zeros(d)
b = 0

#跃迁函数（是神经元是否跳变的关键）
def step(x):
    result = 1 * (x>0)
    return result
#神经元的结果
def y(x):
    z = np.dot(w,x) + b
    a = step(z)
    return a
#训练集的正确结果
def t(i):
    if i < N:
        return 0
    else:
        return 1

print(x1)
print(x2)
#简单感知机的训练模型
while True:
    classified = True
    #遍历数据集开始训练
    for i in range(N * 2):
        print(x[i])
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        #更新权重和偏置量
        w = w + delta_w
        b += delta_b
        #全分类真确后停止训练
        classified *= all(delta_w == 0) * (delta_b == 0)
    if classified:
            break
print("final matrix is")
print(w)
print(b)
print("test Data")
print(y([0,0]))
print(y([5,5]))

