#检测长期依赖信息的训练评估的数据
#生成检测的数据

import numpy as np

class addProblem:
    def __init__(self,T=200):
        self.T = T
    #创造mask的def
    def mask(self):
        #<class 'numpy.ndarray'>
        mask = np.zeros(self.T)
        indices = np.random.permutation(np.arange(self.T))[:2]#np.arange(self.T)生成与mask长度相当的序号序列
        #再使用np.random.permutation()来将序列打乱，同时取出前两个的序号，来吧mask的对应序号的值变为1
        mask[indices] = 1
        return mask

    #生成玩具问题的输入数据和正确结果集的数据
    def toy_problem(self,mask,N=10):
        #(0,1)上的均匀分布
        signal = np.random.uniform(low=0.0,high=1.0,size=(N,self.T))
        masks = np.zeros((N,self.T))
        #masks的每一行的mask的值都是一样的
        for i in range(N):
            #这里我的mask是直接把上一个函数的结果传进来了，就不用再用一次这个函数了
            masks[i] = mask

        data = np.zeros((N,self.T,2))#还是一个三维的输入（多少条（N），每条多少个时间序列（T），每个时间序列有几维的数据（2））
        data[:, :,0] = signal[:]
        data[:, :,0] = masks[:]
        #训练集的Y
        target = (signal * masks).sum(axis=1).reshape((N,1))#axis=1就是将一个矩阵的每一行向量相加

        return (data,target)



if __name__ == '__main__':
    #测试一下是否成功
    model = addProblem(200)
    mask = model.mask()
    data,target = model.toy_problem(mask=mask,N=10)
    print(target)