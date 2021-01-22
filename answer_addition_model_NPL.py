import tensorflow as tf
import numpy as np
# 分割模型的使用的数据
from sklearn.model_selection import train_test_split
# 训练数据的打乱
from sklearn.utils import shuffle
# 每次都要记录一下（ubantu的matplotlib的使用方法有一些差异）
import matplotlib.pyplot as plt

# 随机稳定
np.random.seed(0)
tf.set_random_seed(1234)


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

    # 事先定义输出层（隐藏层到输出层过一个线性层加线性的激活函数）的权重和偏置的shape
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
                # softmax的激活函数
                out = tf.nn.softmax(linear)
                # 保存成模型每一步的输出
                outputs.append(out)
                # 变为独热编码
                # one_hot参数如果索引是标量,则输出形状将是长度 depth 的向量.
                # on_value 和 off_value必须具有匹配的数据类型.如果还提供了 dtype,则它们必须与 dtype 指定的数据类型相同.
                #
                # 如果未提供 on_value,则默认值将为 1,其类型为 dtype.
                # 如果未提供 off_value,则默认值为 0,其类型为 dtype.
                out = tf.one_hot(tf.argmax(out, -1), depth=output_digits)
                (output, decoder_state) = decoder(out, decoder_state)
            # 不管是训练还是其他的流程，都要输出都要保存
            decoder_outputs.append(output)
        # 最后的输出
        if is_training is True:
                # 最后是训练过程的输出时
                # 先将decoder_outputs reshape一下（数据数量， 序列长度， 隐藏层的维度）
                output = tf.reshape(tf.concat(decoder_outputs, axis=1), [-1, output_digits, n_hidden])
                # 使用的时爱英斯坦求和约定
                # shape也会变为[-1, output_digits, n_out]
                linear = tf.einsum('ijk,kl->ijl', output, V) + c
                # 相当于传统的矩阵求和
                # linear = tf.matmul(output, V) + c
                # 最后的输出结果
                return tf.nn.softmax(linear)
        else:
                # 正常使用的最后结果
                # 求最后的输出，线性层加激活函数还是一样的
                linear = tf.matmul(decoder_outputs[-1], V) + c
                out = tf.nn.softmax(linear)
                outputs.append(out)
                # reshape输出的格式
                output = tf.reshape(tf.concat(outputs, axis=1), [-1, output_digits, n_out])
                return output

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
    # argmax()的参数：input：输入Tensor,axis：0表示按列，1表示按行
    # 因为数据的维度是（数据数量， 序列长度， 隐藏层的维度）所以argmax的参数axis维调整为-1
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(yPredict, -1))
    # 最后计算出链表的平均精确度
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

if __name__ == '__main__':
    # 生成数据的条数
    N = 20000
    N_train = int(N * 0.9)
    N_validation = N - N_train
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
        # 使用的时枚举返回的时（index, 字符）
        char_indices = dict((c, i) for i, c in enumerate(chars))
        # 从向量维度到文字的对应关系
        indices_char = dict((i, c) for i, c in enumerate(chars))
        # 独热编码的生成方式
        # 先生成的是零向量的矩阵，再用循环的方式去将独热编码的需要的位数变为1
        # 定义输入(问题数量， 输入|输出位数， 独热编码的长度)
        X = np.zeros((len(question), input_digits, len(chars)), dtype=np.integer)
        # 定义答案
        Y = np.zeros((len(question), digits + 1, len(chars)), dtype=np.integer)
        # 将输入矩阵和答案矩阵，相应位置的0变为一，变为独热编码
        for i in range(N):
            for t, char in enumerate(question[i]):
                X[i, t, char_indices[char]] = 1
            for t, char in enumerate(answer[i]):
                Y[i, t, indices_char[char]] = 1

        # 分割训练集和测试验证集
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, train_size=N_train)

        '''
             模型属性的定义
        '''
        n_in = len(chars) # 这里是0-9和加号和空格
        # LSTM隐藏层的维度
        n_hidden = 128
        n_out = len(chars)
        # 输入的占位符
        # 要传入的几个属性(4个)
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_digits, n_in])
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_digits, n_out])
        n_batch = tf.placeholder(tf.int32, shape=[])
        is_training = tf.placeholder(tf.bool)

        '''
        模型的训练的流程的定义
        '''
        # 训练过程
        yPredict = inference(x=x, y=y, n_batch=n_batch, is_training=is_training, input_digits=input_digits,
                             output_digits=output_digits, n_hidden=n_hidden, n_out=n_out)
        # 误差分析
        loss = loss(y=y, yPredict=yPredict)
        # 优化器
        train_step = training(loss)
        # 精确度的判断
        acc = accuracy(y=y, yPredict=yPredict)

        # 历史数据的保存
        # 验证的误差和验证的精确度（每个迭代的情况）
        history = {
            'val_loss': [],
            'val_acc': []
        }

        '''
        模型的训练
        '''
        epochs = 200
        # 分批次批次的size
        batch_size = 200

        # 会话的老定义
        init = tf.global_variables_initializer()
        sess = tf.Session()
        # 初始化会话
        sess.run(init)

        # 分成的批次的数量
        n_batches = N_train // batch_size

        # 训练过程的可视化
        for epoch in range(epochs):
            print('--' * 10)
            print('epoch', epoch)
            print('--' * 10)

            X_ ,Y_ = shuffle(X_train, Y_train)

            # 批次的训练，为的是使用小批次的梯度下降
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                # 会话开始训练的流程
                sess.run([loss, acc, train_step], feed_dict={x: X_[start, end], y: Y_[start, end], n_batch: batch_size, is_training: True})

            # 使用验证集来测试
            # 每次迭代的loss和acc
            # 实例的eval()方法，调用模型来进行验证
            val_loss = loss.eval(session=sess, feed_dict={
                x: X_validation,
                y: Y_validation,
                n_batch: N_validation,
                is_training: False
            })

            val_acc = acc.eval(session=sess, feed_dict={
                x: X_validation,
                y: Y_validation,
                n_batch: N_validation,
                is_training: False
            })

            # 保存数据，用于matplotlib的画图呈现最后的结果
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            # 每次迭代的显示数据
            print('validation_loss', val_loss)
            print('validation_acc', val_acc)


