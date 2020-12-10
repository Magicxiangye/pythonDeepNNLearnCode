#可视化数据库matplotlib的代码实验
from collections import Counter


from matplotlib import pyplot as plt


# simpleImage
def SimpleImage():
    #两个坐标轴的数据（个数要相同）
    years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    gdp = [100, 200, 345, 674, 343, 349, 567]

    # 线图(颜色、标记点的形状、line的样式)
    plt.plot(years, gdp, color='blue', marker='o', linestyle='solid')
    # 标题
    plt.title('country dream GDP')
    # 给坐标轴添加标记
    plt.ylabel('money')
    plt.show()

#条形图
def lineBar():
    x1 = ['python', 'java', 'php', 'android']
    num_learn = [8,6,1,4]

    #条形的默认宽度维0.8,对左侧的坐标加上0.1
    #使每个条形就可以被放在中心
    xs = [i + 0.1 for i,_ in enumerate(x1)]
    #画图
    plt.bar(xs,num_learn)
    plt.ylabel("学习率")
    plt.title('code learn')

    #x轴的标记(节点的名称改变一下)
    plt.xticks([i + 0.1 for i ,_ in enumerate(x1)],x1)

    plt.show()


#线性图
def labLine():
    variance = [1,2,4,8,16,32,64,128,256]
    bias_squared = [256,128,64,32,16,8,4,2,1]
    total_error = [x+y for x,y in zip(variance,bias_squared)]
    xs = [i for i,_ in enumerate(variance)]#x轴的数据

    #多次调用plt.plot()来在同一张的图上画不同的线
    plt.plot(xs, variance,'g-', label='variance')
    plt.plot(xs, bias_squared, 'r-', label='bias_squared')
    plt.plot(xs, total_error, 'b:', label='total_error')

    #图中各个标识的设置(中文的显示还是有点问题)
    plt.legend(loc=9)#设置的是图中各个label的摆放的位置
    plt.xlabel("模型的复杂度")
    plt.title('test number')
    plt.show()


#散点图
def scatterChart():
    #成对数据集的可视化的好选择
    friend =  [70, 65, 72, 63, 71, 64, 60, 64, 67]
    minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    #使用的画图的工具
    plt.scatter(friend,minutes)

    #散点图的每个对应点的标记
    #还是会用到压缩的函数zip()
    for label, friend_count,minute_count in zip(labels,friend,minutes):
        #每一个点标记的设定
        #xy为被注释的坐标点xytext为注释文字的坐标位置(textcoords有多种的参数)
        plt.annotate(label, xy=(friend_count,minute_count), xytext=(-5,5), textcoords='offset points')

    # 出图前最后的标记
    #plt.axis()可以设置带有比较轴的散点图
    plt.title('friends and play munite relation')
    plt.xlabel('friend_count')
    plt.ylabel('play times')
    plt.show()



if __name__ == '__main__':
    # grades = [77,54,78,89,90,93,97]
    # decile = lambda grade: (grade//10) * 10
    # histest = Counter(decile(grade) for grade in grades)
    # print(histest)
    #labLine()
    scatterChart()


