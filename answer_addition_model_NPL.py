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


# Rnn的编码器-解码器都将使用的是LSTM
def inference(x, y, n_batch, is_training, input_digits=None, n_hidden=None, output_digits=None, n_out=None):
    # 权重变量的函数(老方法)
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # 网络中偏移变量的函数方法
    def bias_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Encoder(编码器)
    #  这个是没有窥视孔功能的LSTM（.LSTMCell()是有窥视孔功能的，参数也是一样用）
    encoder = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias= 1.0)
    # 初始化LSTM的状态
    state = encoder.zero_state(n_batch, tf.float32)
    encoder_outputs = []
    encoder_states = []
    with tf.variable_scope('Encoder'):
        for t in range(input_digits):
            if t > 0:
                # 创建一个新的计算图时候，reuse默认是None，也就是里面参数不可以复用，
                # reuse的作用是复用计算图中的使用的参数
                tf.get_variable_scope().reuse_variables()
            # LSTM的循环网络的训练
            (output, state) = encoder(x[:, t, :], state)
            # 保存一次编码的输出和神经网络的最终的状态
            encoder_outputs.append(output)
            encoder_states.append(state)
    # 接下来是解码器的代码（Decoder）
    # 解码器的LSTM的初始状态相当于数据进入编码器的LSTM的state的最终状态
    decoder = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 从保存的数组中获取相应的网络的最终的状态（这就是解码器的初始的状态）
    decoder_state = encoder_states[-1]
    # 编码器的最终输出，也就是解码器的初始的输出（个人的理解，后期要是学习到觉得不对的话，会进行修改）
    decoder_outputs = [encoder_outputs[-1]]

    # 事先定义输出层的权重和偏置
    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    # 保存输出
    outputs = []

    # 解码器的训练过程
    with tf.variable_scope('Decoder'):
        for t in range(1, output_digits):
            if t > 1:
                # 还是复用作用域里的变量
                tf.get_variable_scope().reuse_variables()
            # 循环神经网络的过程
            # 当是训练的过程的时候
            if is_training is True:
                (output, decoder_state) = decoder(y[:, t - 1, :], decoder_state)
            else:
                # 不是训练的话，作为使用的过程
                # 使用前一个输出作为输入，来进行模型的使用
                # 将会使用爱英斯坦求和约定
                # 不是训练的时候，采用的是线性输出层加偏置来使编码器的输出进入输出层加激活函数，变为真正的模型的输出
                # 来一步步的进入LSTM模型
                # V 和c 都是之前训练时更新好的输出层的权重矩阵和偏移量
                linear = tf.matmul(decoder_outputs[-1], V) + c





# 损失函数还是之前的
# 交叉熵的损失函数
def loss(y, yPredict):
    # tf的几个常用的函数
    # tf.clip_by_value()是使张量的值限定在一个值范围里，可以取等
    # tf.reduce_sum（）的reduction_indices参数，是表示函数的处理维度。
    # 它的参数：[0]:第一维对应的相加，[1], [2]以此类推
    # reduction_indices参数的值默认的时候为None,默认把所有的数据求和，即结果是一维的。
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(yPredict, 1e-10, 1.0)),reduction_indices=[1]))
    return cross_entropy

# 优化器的选取
def training(loss):
    # 选取的是Adam优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.0, beta2=0.999)
    # 最小化的损失
    train_step = optimizer.minimize(loss)
    return train_step

# 精确度的计算
def accuracy(y, yPredict):
    # 每行最大值的索引是否一样(压缩的是第二维)
    # 也就是找每行最大值的下标
    # 返回值correct_prediction 是一个布尔值链表。计算模型的精度还要计算链表的均值。
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(yPredict, -1))
    # 最后计算出链表的平均精确度
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

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
        new_answer = str(a + b)
        # 补充输出的长度
        new_answer = padding(new_answer, output_digits)

        # 保存到相应的数组中
        added.add(pair)
        question.append(new_question)
        answer.append(new_answer)

        # 对数据的独热编码的维度的定义
        # 使用的是枚举的方法
        chars = '0123456789+ '
        # 从文字到向量维度的对应关系
        char_indices = dict((c, i) for i, c in enumerate(chars))
        # 从向量维度到文字的对应关系
        indices_char = dict((i, c) for i, c in enumerate(chars))
        # 独热编码的生成方式
        # 先生成的是零向量的矩阵，再用循环的方式去将独热编码的需要的位数变为1