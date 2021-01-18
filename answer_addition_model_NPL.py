import tensorflow as tf
import numpy as np

# 回答加法计算的回答神经网络
# 生成随机数的函数
# digits默认设置的是最多生成三位数的数
def n(digits=3):
    # 先拿来存放每一位的随机生成的数
    # 最后再用函数来转换为int型
    number = ''
    # 是从序列中获取一个随机元素
    for i in range(np.random.randint(1, digits + 1)):
        number += np.random.choice(list('0123456789'))
    # 转换类型返回数值
    return int(number)



if __name__ == '__main__':
    a, b = n(), n()
    question = '{}+{}'.format(a,b)
    print(question)