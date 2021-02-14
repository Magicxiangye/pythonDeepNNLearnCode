
# MemN记忆网络的问答系统

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
# 画图
import matplotlib.pyplot as plt
#测试常用数据集
# python3.x 使用reduce方法的import
from functools import reduce
# 匹配正则表达式的库
import re

# 数据集的来源
# bAbi任务，Facebook人工智能实验室的数据集
# http://www.thespermwhale.com/jaseweston/babi/
# 需要的文件夹在bAbi_data文件夹

# 填充每个单词到最大长度的字符
def padding(words, maxlens):
    for i, word in enumerate(words):
        words[i] = [0] * (maxlens - len(word)) + word
    # 转型为np数组
    return np.array(words)

# 将单词向量数值化
# 要使用文字到维度的转换数组
# 数据， 单词向量的指标， 故事的最大长度， 问题的最大长度
def vectorize_stories(data, word_indices, story_maxlen, question_maxlen):
    X = []
    Q = []
    A = []
    for story, question, answer in data:
        x = [word_indices[w] for w in story]
        q = [word_indices[w] for w in question]
        # 答案是哪个单词，用的是独热编码的方式来存储
        a =np.zeros(len(word_indices) + 1) # 用于填充，都加1，为的可能是下标的易于辨认
        # 是答案的单词，独热编码
        a[word_indices[answer]] = 1
        X.append(x)
        Q.append(q)
        A.append(a)
    # 最后都要填充到最大的长度才可以输出
    # 独热编码变形为数组
    return (padding(X, maxlens=story_maxlen), padding(Q, maxlens=question_maxlen), np.array(A))


# 使用正则库来切割问题的语句
def tokenize(sent):
    # 列表生成式来简化函数的代码
    # \W匹配任何非单词字符
    # re接收正则表达式，带（）会把用作分割的也保留下来，再先走一遍strip()去掉空格
    final_sent = [x.strip() for x in re.split('(\W+)', sent) if x.strip()]

    return final_sent


# 故事问题的分解组合函数
# 整理为一个问题对应一个故事
# reduce() 函数会对参数序列中元素进行累积。function -- 函数，有两个参数
# iterable -- 可迭代对象
# initializer -- 可选，初始参数
# 将文章分解为单词的形式并进行保存，以便于进行向量化
def get_stories(file, max_length=None):
    # 设置函数内部的展开句子函数
    def flatten(data):
        # 匿名函数来拼接句子的每一个单词
        return reduce(lambda x, y: x + y, data)

    # 开始处理传入的故事文件
    # 先开始分解传入的文件
    data = parse_stories(file.readlines())
    # 合成一个问题对应一个故事并写出解答
    # 上一行data里的每一个数据都是
    # 函数的条件是，不限制长度和故事小于最大的长度
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]

    return data



# 对原始的数据进行分解的函数
def parse_stories(lines):
    data = []
    story = []
    # 每行语句的进行分析
    for line in lines:
        # 解码，strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）
        line = line.strip()
        # 在切割每一行，问题的id号，以及问题的本体(空格，切割一次)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # 每个故事的开始都是id号为1开始的，当切割到id号为一的时候
            # 说明是新的故事和问题的开始，story的数组就要清零
            story = []
        # 当在某一行里有\t，制表符的时候，说明这一行是问题和答案行和支持答案的来源语句的编号，
        # 就要进行提取
        if '\t' in line:
            q, a, supporting = line.split('\t')
            # 将问题分割为单词形式
            # 使用自定义的函数
            q = tokenize(q)
            # 故事的语句
            substory = [x for x in story if x]
            # 集合到数据(以set的形式)
            data.append((substory, q, a))
            # 每一个问句后的分割
            story.append('')
        else:
            # 只是故事的语句，直接切割保存就可以
            sent = tokenize(line)
            story.append(sent)

    return data




# 还是模型的定义
# x:故事；q:问题；vocab_size:词库的大小；embedding_dim:词嵌入的维度
def inference(x, q, n_batch, vocab_size=None, embedding_dim=None, story_maxlen=None, question_maxlen=None):
    def weight_variable(shape):
        # 权重变量的函数
        initial = tf.truncated_normal(shape, stddev=0.08)
        return tf.Variable(initial)
    def height_variable(shape):
        # 偏移量的定义函数
        initial = tf.truncated_normal(shape, stddev=1.0)
        return tf.Variable(initial)

    # 先定义的是用于嵌入的权重矩阵
    # A,C用于故事的输入和输出的词嵌入算法的权重矩阵
    # B是用于问题的词嵌入算法的权重矩阵
    A = weight_variable([vocab_size, embedding_dim])
    B = weight_variable([vocab_size, embedding_dim])
    # 故事的输出的维度为question，因为要和故事的输出添加
    C = weight_variable([vocab_size, question_maxlen])
    # tensorflow自带的嵌入处理的接口
    # 记忆网络的模型化训练，对故事和问题进行嵌入式处理
    # 记忆网络模型显示走的是传统的MemN模型，
    # 在最后一步要输出的时候，加上LSTM模块进行精确度的提升，再走一个前馈神经网络层进行输出
    # 这三个是记忆模型流程中需要的词嵌入的数据
    # m是故事嵌入用于输入的存储
    m = tf.nn.embedding_lookup(A, x)
    # u是输入问题的词嵌入（B是这个算法的词嵌入矩阵）
    u = tf.nn.embedding_lookup(B, q)
    # 用于输出的存储的故事词嵌入
    c = tf.nn.embedding_lookup(C, x)
    # 每个记忆的匹配度
    # 还是使用爱因斯坦的求和定律
    # 这里的问题q是（单词序列）时间序列，要用的是einsum
    # 这样再公式中o的表达式就会发生一些变化
    p = tf.nn.softmax(tf.einsum('ijk,ilk->ijl', m, u))
    # 原来的公式是累乘p与c，得到最后要进入前馈神经层的o
    # 广播机制的相加，按照维度来进行相加，相同的维度保留，其他按广播机制进行扩张
    o = tf.add(p, c)
    # 交换不同维中的内容
    o = tf.transpose(o, perm=[0, 2, 1])
    # 得到的是最后要输出的o和u的集合
    # tf.concat()连接函数(以最后一个维度进行连接)
    ou = tf.concat([o, u], axis=-1)

    # 按照文献的改进的方法，隐藏层到输出层之间，再加上一层LSTM，加强问答的精确度
    cell = tf.contrib.rnn.BasicLSTMCell(embedding_dim//2, forget_bias=1.0)
    # LSTM块的初始状态
    initial_state = cell.zero_state(n_batch, tf.float32)
    state = initial_state
    outputs = []
    # question_maxlen，是以单词为计量单位的计算问题的长度
    with tf.variable_scope('LSTM'):
        for t in range(question_maxlen):
            if t > 0:
                # 复用变量
                tf.get_variable_scope().reuse_variables()
            # 可能再实际中，行列的设置可能会不一样，
            (cell_output, state) = cell(ou[:, t, :], state)
            outputs.append(cell_output)

    # LSTM层到输出层
    output = outputs[-1]
    # 走的是简单的线性层
    W = weight_variable([embedding_dim//2, vocab_size])
    a = tf.nn.softmax(tf.matmul(output, W))
    # 训练的输出
    return a

# 还是多分类的交叉熵损失函数
def loss(y, yPredict):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(yPredict, 1e-10, 1.0)),reduction_indices=[1]))
    return cross_entropy

# 模型的优化器
def training(loss):
    # Adam动量优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    training_step = optimizer.minimize(loss)

    return training_step


# 预测精确度的 验证
def accuracy(y, yPredict):
    # 每行最大值的索引是否一样(压缩的是第二维)
    # 也就是找每行最大值的下标
    # 返回值correct_prediction 是一个布尔值链表。计算模型的精度还要计算链表的均值。
    # argmax()的参数：input：输入Tensor,axis：0表示按列，1表示按行
    # 先得出一个布尔列表的值，在转换为float，算出最后的平均值
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yPredict, 1))
    # 算出最后的精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

if __name__ == '__main__':
    # 开始训练的流程
    # 先是数据的准备阶段
    # 读取训练集和测试集

    file_train = open("bAbi_data/qa1_single-supporting-fact_train.txt", mode="r")
    file_test = open("bAbi_data/qa1_single-supporting-fact_test.txt", mode="r")

    # 分解故事
    train_stories = get_stories(file=file_train)
    test_stories = get_stories(file=file_test)
    # 语料库的set
    vocab = set()
    for story, q, answer in train_stories + test_stories:
        # 按位或赋值(|=),按set计入词汇表
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)
    vocab_size = len(vocab) + 1 # 用于填充1

    # 得出最大的故事长度和问题的长度
    # map() 会根据提供的函数对指定序列做映射。
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    question_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print("向量化数据词汇")
    # 词的坐标，先设置语料库的词汇的位置，再生成独热编码
    word_indices = dict((c, i + 1) for i, c in enumerate(vocab))
    # 把单词向量化，生成独热编码
    inputs_train, question_train, answer_train = \
        vectorize_stories(train_stories, word_indices, story_maxlen, question_maxlen)

    inputs_test, question_test, answer_test = \
        vectorize_stories(test_stories, word_indices, story_maxlen, question_maxlen)

    '''
       模型的生成
    '''
    print("model Building")
    # 先设定占位符placeholder
    # batch, 故事、问题、答案
    x = tf.placeholder(tf.int32, shape=[None, story_maxlen])
    q = tf.placeholder(tf.int32, shape=[None, question_maxlen])
    a = tf.placeholder(tf.float32, shape=[None, vocab_size])
    # 每次分批次，每一批次的个数，LSTM网络都要知道
    n_batch = tf.placeholder(tf.int32, shape=[])

    '''
      训练的定义
    '''
    yPredict = inference(x=x, q=q, n_batch=n_batch,
                         vocab_size=vocab_size, embedding_dim=64,
                         story_maxlen=story_maxlen, question_maxlen=question_maxlen)
    # 误差分析
    loss = loss(y=a, yPredict=yPredict)
    # 优化器
    train_step = training(loss)
    # 精确度的生成
    acc = accuracy(y=a, yPredict=yPredict)
    # 历史记录的保存，用作可视化
    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
       模型的训练
    '''

    print("start model training ")
    # 迭代次数
    epochs = 120
    # 一个批次的大小
    batch_size = 100

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 批次数
    n_batches = len(inputs_train) // batch_size

    # 开始训练
    for epoch in range(epochs):
        # 先把各训练集打乱
        inputs_train_, questions_train_, answers_train_ = \
            shuffle(inputs_train, question_train, answer_train)

        # 每轮都是小批次的梯度下降
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            # 会话
            sess.run(train_step, feed_dict={
                x: inputs_train_[start:end],
                q: questions_train_[start:end],
                a: answers_train_[start:end],
                n_batch: batch_size
            })

        # 使用测试数据进行评估
        val_loss = loss.eval(session=sess, feed_dict={
                x: inputs_test,
                q: question_test,
                a: answer_test,
                n_batch: len(inputs_test)
        })

        # 测试集的精确度
        val_acc = acc.eval(session=sess, feed_dict={
                x: inputs_test,
                q: question_test,
                a: answer_test,
                n_batch: len(inputs_test)
        })

        # 保存
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print('epoch', epoch, 'val_loss', val_loss, 'val_acc', val_acc)

    # 画图
    matlib_loss = history['val_loss']
    matlib_acc = history['val_acc']
    # 字体
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(matlib_loss)), matlib_loss,
                     label='loss', color='black')
    plt.plot(range(len(matlib_acc)), matlib_acc, label='acc', color='red')
    plt.xlabel('epochs')
    plt.show()



