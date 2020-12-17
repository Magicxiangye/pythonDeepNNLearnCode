
#神经网络图像的可视化
import numpy as np
import matplotlib.pyplot as plt
#测试常用数据集
from sklearn import datasets
#应用两个工具
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    # MNIST打乱顺序抽取出N数据来进行测试
    n = len(mnist.data)
    N = 3000  # 选取部分MNIST数据进行实验
    N_train = 2000
    N_validation = 400  # 选取1000个来当神经网络的数据
    # 取出的index的list
    indices = np.random.permutation(range(n))[:N]

    # 准备数据（先做近似min-max的归一化处理（公式在书上））
    X = mnist.data[indices]
    X = X / 255.0
    X = X - X.mean(axis=1).reshape(len(X), 1)
    # 做三维的数据
    X = X.reshape(len(X), 28, 28)  # 转换为时间序列数据
    # 真实集
    Y = mnist.target[indices]
    Y = np.eye(10)[Y.astype(int)]  # 转换为1-of-K形式

    # 划分一下训练和测试的数据
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=N_validation)

    fig, ax = plt.subplots(nrows=10, ncols=10, sharex='all', sharey='all')
    ax = ax.flatten()
    for i in range(100):
        img = X_train[i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()