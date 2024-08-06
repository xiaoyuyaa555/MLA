import random
import numpy as np


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
    return weights


def stocGradAscent1(data_mata, class_labels, num_iter=150):
    m, n = np.shape(data_mata)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))  # 数据集的索引列表
        for i in range(m):
            alpha = 4 / (1 + j + i) + 0.03  # 动态学习率
            rand_index = int(random.uniform(0, len(data_index)))  # 随机选取样本
            h = sigmoid(sum(data_mata[data_index[rand_index]] * weights))  # 选择随机选取的一个样本，计算h
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mata[data_index[rand_index]]
            del (data_index[rand_index])  # 删除用过的索引
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    f_train = open('./horseColicTraining.txt')
    f_test = open('./horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in f_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stocGradAscent1(np.array(training_set), training_labels, 500)
    error = 0
    num_test_vec = 0.0
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classifyVector(line_arr, train_weights)) != float(curr_line[21]):
            error += 1
    rate_err = (error / num_test_vec) * 100
    # print("错误率为：%.2f%%" % rate_err)
    return rate_err


def multiTest(num_test=20):
    error = 0.0
    for i in range(num_test):
        error += colicTest()
    print("%d次测试总的错误率是：%.2f%%" % (num_test, error / num_test))


if __name__ == '__main__':
    multiTest()



