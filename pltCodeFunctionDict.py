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

if __name__ == '__main__':
    grades = [77,54,78,89,90,93,97]
    decile = lambda grade: (grade//10) * 10
    histest = Counter(decile(grade) for grade in grades)
    print(histest)


