from math import inf

import numpy as np


def load_simple_data():
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])

    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return data_mat, class_labels


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    """
    :param data_mat:    输入的数据矩阵
    :param dimen:       选择的特征的索引
    :param thresh_val:  分类阈值
    :param thresh_ineq: 阈值的比较运算符，"lt"小于等于,"gt"大于
    :return:分类结果数组
    """
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, d_vector):
    """
    :param data_arr:        输入的数据数组
    :param class_labels:    输入样本的标签数组
    :param d_vector:        样本权重的向量
    :return:                详细信息 best_stump、最小错误率 min_error和最佳分类结果 best_class
    """
    data_mat_in = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat_in)
    num_steps = 10.0
    best_stump = {}
    best_class = np.mat(np.zeros((m, 1)))

    min_error = inf
    # 对样本的每个特征
    for i in range(n):
        range_min = data_mat_in[:, i].min()
        range_max = data_mat_in[:, i].max()
        step_size = (range_max - range_min) / num_steps
        # 对每个步长
        for j in range(-1, int(num_steps) + 1):
            # 对每个不等号
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat_in, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                # 降低分类正确的样本的权重，将分类正确的设置错误权重为0，错误的为1
                err_arr[predicted_vals == label_mat] = 0
                # 计算加权错误率
                weighted_err = (d_vector.T * err_arr)
                # print(weighted_err)
                # print("split:dim%d,thresh: %0.2f,thresh inequal:%s,the weighted error is %0.3f" % (i,thresh_val,inequal,weighted_err))
                # 如果当前的加权错误率较小，则当前分类为最佳分类best_class，更新min_error的值和best_step字典
                if weighted_err < min_error:
                    min_error = weighted_err
                    best_class = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    # print("best_stump=%s" % best_stump)
    return best_stump, min_error, best_class


def ada_boost_train(data_mat_in, class_labels_in, num_iter=40):
    """

    :param data_mat_in:         输入的数据矩阵
    :param class_labels_in:     输入样本的标签数组
    :param num_iter:            分类器的数目，默认40
    :return:                    弱分类器列表
    """
    # 弱分类器列表
    weak_classifier_list = []
    data_mat_in = np.mat(data_mat_in)
    m, n = data_mat_in.shape
    # 每个样本的权重向量
    d_vector = np.mat(np.ones((m, 1)) / m)
    # 累积预测结果
    agg_class_est = np.mat(np.zeros((m, 1)))

    for _ in range(num_iter):
        # 寻找最佳的单层决策树
        stump, error, class_est = build_stump(data_mat_in, class_labels_in, d_vector)
        # 计算权重值alpha
        alpha = float(0.5 * np.log((1.0 - error) / error))  # 使用 max(error, 1e-16) 是为了避免发生除零错误
        stump['alpha'] = alpha
        weak_classifier_list.append(stump)
        # 更新权重向量D
        expon = np.multiply(-1 * alpha * np.mat(class_labels_in).T, class_est)
        d_vector = np.multiply(d_vector, np.exp(expon))
        d_vector = d_vector / d_vector.sum()

        # 分类的结果是各分类器权重的加权和
        agg_class_est += alpha * class_est
        # 通过与真实类别标签 class_labels 比较，生成一个布尔值的数组，表示哪些样本被错误分类
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels_in).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m

        # print('class_est:', class_est.T)
        # print('agg_class_est:', agg_class_est)
        # print('total error:', error_rate, '\n')
        if error_rate == 0.0:
            break
    return weak_classifier_list, agg_class_est


def ada_classify(data_in, weak_classifier_list):
    data_mat_in = np.mat(data_in)
    m, n = data_mat_in.shape
    # 累积预测结果
    agg_class_est = np.mat(np.zeros((m, 1)))

    # 对每个弱分类器
    for classifier in weak_classifier_list:
        # 得到当前分类器的分类预估值
        class_est = stump_classify(data_mat_in, classifier['dim'], classifier['thresh'], classifier['ineq'])
        # 得到分类结果
        agg_class_est += classifier['alpha'] * class_est
        # print(agg_class_est)

    return np.sign(agg_class_est)


def plotROC(pred_strengths, class_labels):
    import matplotlib.pyplot as plt
    # 保存绘制光标的位置
    cur = (1.0, 1.0)
    # 用于计算AUC的值
    y_sum = 0.0
    num_pos_clas = np.sum(np.array(class_labels) == 1.0)
    # 得到步长
    y_step = 1 / float(num_pos_clas)
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    sorted_indices = pred_strengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:
        if class_labels[index] == 1.0:  # 真阳
            del_x = 0
            del_y = y_step
        else:                           # 假阳
            del_x = x_step
            del_y = 0
            # 对小矩形面积进行累加
            y_sum += cur[1]
        # 绘制实线
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    # 绘制虚线
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("AUC指数:", y_sum * x_step)


def ada_test(test_data_mat, data_label, classifier_list):
    test_data_mat = np.mat(test_data_mat)
    m, n = test_data_mat.shape
    label_est = ada_classify(test_data_mat, classifier_list)

    err_arr = np.ones((m, 1))
    err_rate = err_arr[label_est != np.mat(data_label).T].sum() / m

    return err_rate


if __name__ == '__main__':
    data_mat, class_labels = load_simple_data()
    weight_vector = np.mat(np.ones((5, 1)) / 5)
    build_stump(data_mat, class_labels, weight_vector)
    weak_class_array, a = ada_boost_train(data_mat, class_labels, 9)
