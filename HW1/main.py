import numpy as np
import pandas as pd


# step1 读入数据，预处理
def data_handle():
    # 默认参数 sep 分隔符是逗号
    df = pd.read_csv('train.csv', usecols=range(3, 27), encoding='big5')
    # print(df.head())
    df = df.replace('NR', 0.0)
    # print(df.head(n=18))
    x_list = []
    y_list = []

    array = np.array(df).astype(float)
    # 类似于滑动窗口，进阶版的是把第二天一点的拼接到第一天24点后面
    for i in range(0, 4320, 18):
        for j in range(24-9):
            mat = array[i:i + 18, j:j + 9]
            label = array[i + 9, j + 9]  # 第10行是PM2.5
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)

    # print(x.shape)
    # print(y.shape)
    return x, y


# 求梯度-> 更新 w 和 b， 求 Loss
def train(epoch, x_train, y_train):
    bias = 0  # 偏差初始化
    weights = np.ones(9)  # 权重初始化 是9组 wi
    learning_rate = 1  # 初始化学习率
    reg_rate = 0.001  # 初始化正则系数

    # 用来存梯度平方和，在更新学习率 adagrad 算法中有用到
    w_gradient_sum = np.zeros(9)
    b_gradient_sum = 0

    # 求梯度 b_gradient 和 w_gradient
    for index in range(epoch):  # 训练次数
        b_gradient = 0  # 每次训练初始化
        w_gradient = np.zeros(9)
        for i in range(3200):  # 遍历 3200 个数据集，每个为 18*9 的矩阵
            b_gradient += (y_train[i] - weights.dot(x_train[i, 9, :]) - bias) * (-1)  # dot 是向量相乘，得一个数值
            for j in range(9):  # w 是多维的
                w_gradient[j] += (y_train[i] - weights.dot(x_train[i, 9, :]) - bias) * (-x_train[i, 9, j])

        # 继续算 b_gradient 和 w_gradient 上面算了求和
        b_gradient /= 3200
        w_gradient /= 3200  # 注意 w_gradient 是一个 9 维向量

        # w_gradient 加上正则项
        for i in range(9):
            w_gradient += reg_rate * weights[i]

        # adagrad 算法 更新学习率
        b_gradient_sum += b_gradient ** 2
        w_gradient_sum += w_gradient ** 2  # 向量相加

        # 梯度下降 更新 bias 和 weights
        bias = bias - learning_rate/b_gradient_sum ** 0.5 * b_gradient
        weights = weights - learning_rate/w_gradient_sum ** 0.5 * w_gradient

        # 每训练200轮，输出一次在训练集上的损失
        if index % 200 == 0:
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) ** 2
            print('after {} epochs, the loss on train data is:'.format(index), loss / 3200)

    return weights, bias


def main():
    x, y = data_handle()
    epoch = 2000  # 训练次数
    # 划分训练集与验证集
    x_train, y_train = x[0:3200], y[0:3200]
    x_val, y_val = x[3200:3600], y[3200:3600]

    w, b = train(epoch, x_train, y_train)

    print("the final result: ", w, b)
    print("finished")


if __name__ == '__main__':
    main()
