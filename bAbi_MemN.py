# MemN记忆网络的问答系统

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

# 数据集的来源
# bAbi任务，Facebook人工智能实验室的数据集
# http://www.thespermwhale.com/jaseweston/babi/
# 需要的文件夹在bAbi_data文件夹

# 填充每个单词到最大长度的字符
def padding(words, maxlens):
    for i, word in enumerate(words):
        words[i] = [0] * (len(word) - maxlens) + word
    # 转型为np数组
    return np.array(words)

# 向量再转化为故事
# 要使用文字到维度的转换数组
def vectorize_stories():
    pass

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
    # A,C用于故事的输出和输入的词嵌入算法的权重矩阵
    # B是用于问题的词嵌入算法的权重矩阵
    A = weight_variable([vocab_size, embedding_dim])
    B = weight_variable([vocab_size, embedding_dim])
    C = weight_variable([vocab_size, embedding_dim])
    # tensorflow自带的嵌入处理的接口
    # 记忆网络的模型化训练，对故事和问题进行嵌入式处理
    # 记忆网络模型显示走的是传统的MemN模型，
    # 在最后一步要输出的时候，加上LSTM模块进行精确度的提升，再走一个前馈神经网络层进行输出

if __name__ == '__main__':
    pass