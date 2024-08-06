import numpy as np
from svm_simple import selectJrand, clipAlpha

def loadDataSet(filename):
    data_mat = []
    label_mat = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip().split('\t')
        data_mat.append([float(line[0]), float(line[1])])
        label_mat.append(int(float(line[-1])))
    return data_mat, label_mat


class optStruct:
    def __init__(self, data_mat_in, class_labels, C, toler):
        self.x = data_mat_in
        self.label_mat = class_labels
        self.C = C
        self.toler = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # e_cache用于缓存误差，第一列是标志位，取值为0或者1，为1时表示已经算出来；第二列是实际的E值
        self.e_cache = np.mat(np.zeros((self.m, 2)))


# 计算E值
def calcEk(opt, k):
    f_xk = float(np.multiply(opt.alphas, opt.label_mat).T *
                 (opt.x * opt.x[k, :].T)) + opt.b
    ek = f_xk - float(opt.label_mat[k])
    return ek


# 用于选择第二个α值，返回索引和误差值
def selectJ(i, opt: optStruct, ei):
    max_k = -1
    max_delta_e = 0
    ej = 0
    opt.e_cache[i] = [1, ei]
    valid_ecache_list = np.nonzero(opt.e_cache[:, 0].A)[0]  # 获取有效值的行索引
    # 从现有的有效误差中寻找第二个alpha
    if len(valid_ecache_list) > 1:
        # 遍历，找到最大的Ek
        for k in valid_ecache_list:
            if k == i:
                continue
            ek = calcEk(opt, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        # 如果现有有效误差只有1个（即第一个α），则说明这是第一次循环，直接在所有数据里面进行随机选择
        j = selectJrand(i, opt.m)
        ej = calcEk(opt, j)
    return j, ej


# 计算误差值，并存入缓存
def updateEk(opt, k):
    ek = calcEk(opt, k)
    opt.e_cache[k] = [1, ek]


# 内循环
def innerL(i, opt: optStruct):
    ei = calcEk(opt, i)
    if ((opt.label_mat[i] * ei < -opt.toler) and (opt.alphas[i] < opt.C)) or \
            ((opt.label_mat[i] * ei > opt.toler) and (opt.alphas[i] > 0)):
        j, ej = selectJ(i, opt, ei)
        alpha_i = opt.alphas[i].copy()
        alpha_j = opt.alphas[j].copy()
        if opt.label_mat[i] != opt.label_mat[j]:
            L = max(0, opt.alphas[j] - opt.alphas[i])
            H = min(opt.C, opt.C + opt.alphas[j] - opt.alphas[i])
        else:
            L = max(0, opt.alphas[i] + opt.alphas[j] - opt.C)
            H = min(opt.C, opt.alphas[i] + opt.alphas[j])
        if L == H:
            print("L == H")
            return 0
        # eta是alpha[j]的最优修改量
        eta = 2.0 * opt.x[i, :] * opt.x[j, :].T - opt.x[i, :] * opt.x[i, :].T - opt.x[j, :] * opt.x[j, :].T
        # eta >= 0的情况比较少，并且优化过程计算复杂，所以此处做了简化处理，直接跳过了
        if eta >= 0:
            print("eta >= 0")
            return 0
        opt.alphas[j] -= opt.label_mat[j] * (ei - ej) / eta
        opt.alphas[j] = clipAlpha(opt.alphas[j], H, L)
        updateEk(opt, j)
        # 比对原值，看变化是否明显，如果优化并不明显则退出
        if abs(opt.alphas[j] - alpha_j) <  0.00001:
            print("j not moving enough")
            return 0
        opt.alphas[i] += opt.label_mat[j] * opt.label_mat[i] * (alpha_j - opt.alphas[j])
        updateEk(opt, i)
        b1 = opt.b - ei - opt.label_mat[i] * (opt.alphas[i] - alpha_i) * opt.x[i, :] * opt.x[i, :].T \
             - opt.label_mat[j] * (opt.alphas[j] - alpha_j) * opt.x[i, :] * opt.x[j, :].T
        b2 = opt.b - ej - opt.label_mat[i] * (opt.alphas[i] - alpha_i) * opt.x[i, :] * opt.x[j, :].T \
             - opt.label_mat[j] * (opt.alphas[j] - alpha_j) * opt.x[j, :] * opt.x[j, :].T
        if (0 < opt.alphas[i]) and (opt.C > opt.alphas[i]):
            opt.b = b1
        elif (0 < opt.alphas[j]) and (opt.C > opt.alphas[j]):
            opt.b = b2
        else:
            opt.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


'''
外循环函数
参数：数据集，类别标签，松弛变量C，容错率， 总共最大的循环次数，核函数
C表示不同优化问题的权重，如果C很大，分类器会力图通过分离超平面将样例都正确区分，如果C很小又很难很好的分类，C值需要平衡两者
'''


def smoP(data_mat_in, class_labels, C, toler, max_iter):
    opt: optStruct = optStruct(np.mat(data_mat_in), np.mat(class_labels).transpose(), C, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    # 循环条件1：iter未超过最大迭代次数
    # 循环条件2：上次循环有收获或者遍历方式为遍历所有值
    # 因为最终的遍历方式是趋向于遍历非边界值，如果仍在遍历所有值说明还需要更多的训练
    while iter < max_iter and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        # 遍历所有的值
        if entire_set:
            for i in range(opt.m):
                alpha_pairs_changed += innerL(i, opt)
                print(f"full_set, iter:{iter} i: {i}, pairs changed: {alpha_pairs_changed}")
            iter += 1
        # 遍历非边界值
        else:
            # 取出非边界值的行索引
            non_bound_i_s = np.nonzero(np.array((opt.alphas.A > 0) * (opt.alphas.A < opt.C)))[0]
            for i in non_bound_i_s:
                alpha_pairs_changed += innerL(i, opt)
                print(f"non-bound, iter:{iter} i: {i}, pairs changed: {alpha_pairs_changed}")
            iter += 1
        # 如果上次循环是遍历所有值，那么下次循环改为遍历非边界值
        if entire_set:
            entire_set = False
        # 如果上次循环是遍历非边界值但没有收获，则改为遍历所有值
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number: ", iter)

    return opt.b, opt.alphas


def plotSupport(data_mat, label_mat, supp_points, ws, b):
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
    for i in range(len(supp_points)):
        x, y = supp_points[i][0]
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
    X = np.mat(data_arr)
    labelMat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    ws = np.zeros((n, 1))
    for i in range(m):
        # 大部分的alpha为0，少数几个是值不为0的支持向量
        ws += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return ws


def getSupportPoints(data_mat, label_mat, alphas):
    support_points_list = []
    for i in range(len(label_mat)):
        if alphas[i] > 0.0:
            support_points_list.append([data_mat[i], label_mat[i]])
    return support_points_list


if __name__ == '__main__':
    data_arr, class_labels = loadDataSet('./testSet.txt')
    t_b, t_alphas = smoP(data_arr, class_labels, C=0.6, toler=0.001, max_iter=40)
    print(t_b)
    print(t_alphas[t_alphas > 0])

    w = calcWs(t_alphas, data_arr, class_labels)
    support_points = getSupportPoints(data_arr, class_labels, t_alphas)
    plotSupport(data_arr, class_labels, support_points, w, t_b)
