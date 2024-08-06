import random
import numpy as np


def loadDataSet():
    data_mat = []
    label_mat = []
    f = open('./testSet.txt', 'r')
    for line in f.readlines():
        line = line.strip().split()
        data_mat.append([1.0, float(line[0]), float(line[1])])
        label_mat.append(int(line[2]))
    return data_mat, label_mat


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def gradAscent(data_mat_in, class_labels):
    data_mat = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()  # 取转置
    m, n = np.shape(data_mat)
    alpha = 0.001  # 步进大小
    max_cycles = 500  # 退出条件，迭代500次
    weights = np.ones((n, 1))  # 回归系数初始化为1
    weights_arr = np.array([])
    for i in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = (label_mat - h)  # 实际类别与预测类别的差值
        weights = weights + alpha * data_mat.transpose() * error
        weights_arr = np.append(weights_arr, weights)
    weights_arr = weights_arr.reshape(max_cycles, n)
    return weights.getA(), weights_arr


def stocGradAscent0(data_mata, class_labels, num_iter=150):
    m, n = np.shape(data_mata)
    alpha = 0.01
    weights = np.ones(n)  # 权重初始化为一维数组
    weights_arr = np.array([])
    for j in range(num_iter):
        for i in range(m):
            h = sigmoid(sum(data_mata[i] * weights))  # 每次只取一个样本进行数组计算
            error = class_labels[i] - h
            weights = weights + alpha * error * data_mata[i]  # 矩阵的运算转为数组的运算
            weights_arr = np.append(weights_arr, weights)
    weights_arr = weights_arr.reshape(num_iter * m, n)
    return weights, weights_arr


def stocGradAscent1(data_mata, class_labels, num_iter=150):
    m, n = np.shape(data_mata)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(num_iter):
        data_index = list(range(m))  # 数据集的索引列表
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 动态学习率
            rand_index = int(random.uniform(0, len(data_index)))  # 随机选取样本
            h = sigmoid(sum(data_mata[data_index[rand_index]] * weights))  # 选择随机选取的一个样本，计算h
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mata[data_index[rand_index]]
            weights_array = np.append(weights_array, weights)
            del (data_index[rand_index])  # 删除用过的索引
    weights_array = weights_array.reshape(num_iter * m, n)
    return weights, weights_array


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():

    training_set = []
    training_labels = []
    with open('./horseColicTraining.txt') as f_train:
        for line in f_train.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            training_set.append(line_arr)
            training_labels.append(float(curr_line[21]))
    train_weights = stocGradAscent1(np.array(training_set), training_labels, num_iter=500)
    error = 0
    num_test_vec = 0.0
    with open('./horseColicTest.txt') as f_test:
        for line in f_test.readlines():
            num_test_vec += 1
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            if int(classifyVector(line_arr, train_weights)) != curr_line[-1]:
                error += 1
    rate_err = error / num_test_vec
    print(f"错误率为：{rate_err}")
    return rate_err


def multiTest(num_test=10):
    error = 0.0
    for i in range(num_test):
        this_rate = colicTest()
        error += this_rate
        print(f"第{i}次的错误率为：{this_rate}")
    print(f"{num_test}次测试总的错误率是：{this_rate / num_test}")


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = loadDataSet()
    data_arr = np.array(data_mat)
    m = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(label_mat[i]) == 1:  # 类别为1
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  # 直线方程为 w0 + w1*x1 + w2*x2 = 0，w0是偏移量
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotWeights(weights_array1, weights_array2, weights_array3):
    import matplotlib.pyplot as plt
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法', fontproperties='SimHei', fontsize=12)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', fontproperties='SimHei', fontsize=12)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', fontproperties='SimHei', fontsize=12)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=12)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', fontproperties='SimHei', fontsize=12)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'随机梯度上升算法', fontproperties='SimHei',
                                          fontsize=12)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=12)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x3 = np.arange(0, len(weights_array3), 1)
    # 绘制w0与迭代次数的关系
    axs[0][2].plot(x3, weights_array3[:, 0])
    axs0_title_text = axs[0][2].set_title(u'改进的随机梯度上升算法', fontproperties='SimHei',
                                          fontsize=12)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][2].plot(x3, weights_array3[:, 1])
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][2].plot(x3, weights_array3[:, 2])
    axs2_xlabel_text = axs[2][2].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=12)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    # data_mat1, label_mat1 = loadDataSet()
    # weight0, weight_arr0 = gradAscent(np.array(data_mat1), label_mat1)
    # weight1, weight_arr1 = stocGradAscent0(np.array(data_mat1), label_mat1)
    # weight2, weight_arr2 = stocGradAscent1(np.array(data_mat1), label_mat1)
    # # plotBestFit(weight_arr)
    # plotWeights(weight_arr0, weight_arr1, weight_arr2)
    colicTest()
