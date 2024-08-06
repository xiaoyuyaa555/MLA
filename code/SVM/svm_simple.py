import random
import numpy as np


def loadDataSet(filename):
    data_mat = []
    label_mat = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip().split('\t')
        data_mat.append([float(line[0]), float(line[1])])
        label_mat.append(int(float(line[-1])))
    return data_mat, label_mat


"""
功能：随机选择另一个α
输入：当前α的下标，α的数目
输出：另一个α的下标
"""
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


"""
功能：限制α的大小
输入：α，上限，下限
输出：α
"""
def clipAlpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


"""
输入：数据集，标签列表，常数C，容错率，最大循环次数
"""
def smoSimple(data_mat, class_label, c, toler, max_iter):
    # 将数据和标签列表转换成矩阵
    data_mat = np.mat(data_mat)
    label_mat = np.mat(class_label).transpose()
    b = 0
    m, n = np.shape(data_mat)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:  # 大循环，控制循环次数
        alpha_pairs_changed = 0
        for i in range(m):  # 小循环，优化每一个α
            # 预测类别
            f_xi = float(np.multiply(alphas, label_mat).T *
                         (data_mat * data_mat[i, :].T)) + b
            # 预测误差
            ei = f_xi - float(label_mat[i])
            # 如果误差过大( < -toler 或者 > toler)且 α 满足 0 < α < c 的约束条件,对 α 进行优化.
            if ((label_mat[i] * ei < -toler) and (alphas[i] < c)) or \
                    ((label_mat[i] * ei) > toler and (alphas[i] > 0)):
                # 随机选择另一个样本
                j = selectJrand(i, m)
                # 预测类别
                f_xj = float(np.multiply(alphas, label_mat).T *
                             (data_mat * data_mat[j, :].T)) + b
                # 预测误差
                ej = f_xj - float(label_mat[j])
                # 保存旧值，方便后续做比较
                alpha_i = alphas[i].copy()
                alpha_j = alphas[j].copy()
                # 保证αj满足约束条件
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                if l == h:
                    print('L == H')
                    continue
                # eta是alpha[j]的最佳修正量
                eta = 2.0 * data_mat[i, :] * data_mat[j,].T - \
                        data_mat[i, :] * data_mat[i,].T - \
                        data_mat[j, :] * data_mat[j,].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                # 获得alpha[j]的优化值
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                alphas[j] = clipAlpha(alphas[j], h, l)
                # 对比原值，是否优化明显，不明显则退出
                if abs(alphas[j]) - alpha_j < 0.00001:
                    print('j not moving enough')
                    continue
                # 对alpha[i]、alpha[j]进行量相同、方向相反的优化
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i) * \
                     data_mat[i, :] * data_mat[i,].T - \
                     label_mat[j] * (alphas[j] - alpha_j) * \
                     data_mat[i, :] * data_mat[j, :].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i) * \
                     data_mat[i, :] * data_mat[j,].T - \
                     label_mat[j] * (alphas[j] - alpha_j) * \
                     data_mat[j, :] * data_mat[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print(f"iter:{iter} i:{i}, pairs changed {alpha_pairs_changed}")
        """
        如果所有向量都没有优化，则增加迭代数目，继续下一次循环
        当所有的向量都不再优化，并且达到最大迭代次数才退出，一旦有向量优化，iter都要归零
        """
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print(f"iteration number:{iter}")
    return b, alphas


def plotDataSet():
    import matplotlib.pyplot as plt
    data_mat, label_mat = loadDataSet('./testSetRBF2.txt')
    data_arr = np.array(data_mat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    m = len(label_mat)
    for i in range(m):
        if label_mat[i] == 1:
            xcord1.append(data_arr[i, 0])
            ycord1.append(data_arr[i, 1])
        else:
            xcord2.append(data_arr[i, 0])
            ycord2.append(data_arr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    plt.show()


# 画图并圈出支持向量的数据点
def plotSupport(data_mat, label_mat, support_points, ws, b):
    import matplotlib.pyplot as plt
    data_arr = np.array(data_mat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    m = len(label_mat)
    for i in range(m):
        if label_mat[i] == 1:
            xcord1.append(data_arr[i, 0])
            ycord1.append(data_arr[i, 1])
        else:
            xcord2.append(data_arr[i, 0])
            ycord2.append(data_arr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')

    # 圈出支持向量
    for i in range(len(support_points)):
        x, y = support_points[i][0]
        plt.scatter([x], [y], s=150, c='none', alpha=.5, linewidth=1.5, edgecolor='red')

    # 绘制分离超平面
    x1 = max(data_mat)[0]
    x2 = min(data_mat)[0]
    b = float(b)
    a1 = float(ws[0][0])
    a2 = float(ws[1][0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    ax.grid(True)
    plt.show()


def calcWs(alphas, data_arr, class_labels):
    x = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(x)
    ws = np.zeros((n, 1))
    for i in range(m):
        # 大部分的alpha为0，少数几个是值不为0的支持向量
        ws += np.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return ws


def getSupportPoints(data_mat, label_mat, alphas):
    support_points_list = []
    for i in range(100):
        if alphas[i] > 0.0:
            support_points_list.append([data_mat[i], label_mat[i]])
    return support_points_list


if __name__ == '__main__':
    # data_matx, label_matx = loadDataSet('./testSet.txt')
    # t_b, t_alphas = smoSimple(data_matx, label_matx, c=0.6, toler=0.001, max_iter=40)
    # print(t_b)
    # print(t_alphas[t_alphas > 0])
    #
    # w = calcWs(t_alphas, data_matx, label_matx)
    # support_points = getSupportPoints(data_matx, label_matx, t_alphas)
    # plotSupport(data_matx, label_matx, support_points, w, t_b)
    plotDataSet()
