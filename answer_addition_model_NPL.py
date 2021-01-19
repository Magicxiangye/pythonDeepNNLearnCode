import tensorflow as tf
import numpy as np

# 生成加法问答系统的训练数据，（简单版本）
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

# 为了使加法的每个的向量的长度变得相似，要把各向量的长度变得相同
# 用空格来填充
# 都是以最大的数据的长度来体现
# 先对计算的算式进行填充，最大的输入长度和生成数据的位数有关
def padding(chars, maxlen):
    return chars + ' ' * (maxlen - len(chars))
    # 用空格来进行输入数据的填充

if __name__ == '__main__':
    # 生成数据的条数
    N = 20000
    # 设置一下数据的输入的最大的位数
    digits = 3
    # 同时获取一下输出的最大的位数
    output_digits = digits * 2
    a, b = n(), n()
    # 获取最大的输入的数据位数，以获取最大的输入的长度
    input_digits = digits * 2 +1
    # 用于存储生成的数据
    added = set()
    # 存储输入数据的数组
    question = []
    # 存储输出数据的数组
    answer = []
    # 开始生成数据
    while len(question) < N:
        # 随机生成两个数
        a, b = n(), n()
        # 每一对设置元组的类型(同时使用sorted来进行数据的排序，方便查看)
        pair = tuple(sorted((a, b)))
        # 检验当前的两个数的算式是否在保存数据的set中
        # 存在的话就继续生成其他的加法数据
        if pair in added:
            continue
        # 不存在的话，进行长度的填充后进行存储
        new_question = '{}+{}'.format(a, b)
        # 填充输入数据的长度到一致的长度
        new_question = padding(new_question, input_digits)