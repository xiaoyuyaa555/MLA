import numpy as np
from bs4 import BeautifulSoup
import time
import urllib.request
import json


def load_data_set(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    f = open(file_name)

    for line in f.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))

    f.close()
    return np.array(data_mat), np.array(label_mat)


# 计算最佳拟合直线
def stand_regress(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    xTx = x_mat.T * x_mat

    #  如果行列式的值为0，则不可逆,无法求w
    if np.linalg.det(xTx) == 0.0:
        print("矩阵不可逆")
        return

    ws = xTx.I * (x_mat.T * y_mat)

    return ws


# 局部加权线性回归函数LWLR
def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    m, n = x_mat.shape
    weights = np.mat(np.eye(m))  # eye(m)用于创建一个m*m的对角矩阵，对角线上值全为1

    for i in range(m):
        diff_mat = test_point - x_mat[i, :]
        weights[i, i] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))  # |diff_mat| = diff_mat*diff_mat.T

    # 计算拟合直线
    xTx = x_mat.T * (weights * x_mat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵不可逆")
        return

    ws = xTx.I * (x_mat.T * (weights * y_mat))
    y_hat = test_point * ws
    return y_hat


# 对每个点进行估计
def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = len(test_arr)
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


# 岭回归
def ridge_regress(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T * x_mat
    # denom = xTx + λI
    denom = xTx + np.eye(np.shape(x_mat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵不可逆")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T

    # 计算Y的均值
    y_mean = np.mean(y_mat, 0)
    # 将 y 向量中的每个样本减去均值，消除截距项的影响
    y_mat = y_mat - y_mean

    # # 标准化 x，计算x_mat平均值
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    # 对自变量进行标准化
    x_mat = (x_mat - x_means) / x_var

    # 初始化权重矩阵30*n
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))

    # 循环计算不同λ下的权重系数
    for i in range(num_test_pts):
        ws = ridge_regress(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    in_mat = x_mat.copy()
    in_means = np.mean(in_mat, 0)
    in_var = np.var(in_mat, 0)
    in_mat = (in_mat - in_means) / in_var
    return in_mat


# 逐步线性回归
def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean

    # 对 x 矩阵进行正则化处理
    x_mat = regularize(x_mat)
    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(num_it):
        print(ws.T)
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                # 向两个方向微调回归系数
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                # 计算预测值与实际值之间的误差
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


# 从页面读取数据，生成retX和retY列表
def scrape_page(retX, retY, in_file, yr, num_pce, orig_prc):
    # 打开并读取HTML文件
    fr = open(in_file, encoding='utf-8')
    soup = BeautifulSoup(fr.read(), 'html.parser')
    i = 1

    # 根据HTML页面结构进行解析
    current_row = soup.findAll('table', r="%d" % i)
    while len(current_row) != 0:
        current_row = soup.findAll('table', r="%d" % i)
        title = current_row[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            new_flag = 1.0
        else:
            new_flag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        sold_unicde = current_row[0].findAll('td')[3].findAll('span')
        if len(sold_unicde) == 0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = current_row[0].findAll('td')[4]
            price_str = soldPrice.text
            price_str = price_str.replace('$', '')  # strips out $
            price_str = price_str.replace(',', '')  # strips out ,
            if len(soldPrice) > 1:
                price_str = price_str.replace('Free shipping', '')
            selling_price = float(price_str)

            # 去掉不完整的套装价格
            if selling_price > orig_prc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, num_pce, new_flag, orig_prc, selling_price))
                retX.append([yr, num_pce, new_flag, orig_prc])
                retY.append(selling_price)
        i += 1
        current_row = soup.findAll('table', r="%d" % i)


# 设置数据集的信息
# 依次读取六种乐高套装的数据，并生成数据矩阵
def set_data_collect(retX, retY):
    scrape_page(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)
    scrape_page(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)
    scrape_page(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)
    scrape_page(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)
    scrape_page(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)
    scrape_page(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)


# 交叉验证函数
def cross_validation(x_arr, y_arr, num_val=10):
    # 获得数据点个数，x_arr和y_arr具有相同长度
    m = len(y_arr)
    index_list = list(range(m))
    error_mat = np.zeros((num_val, 30))

    # 主循环 交叉验证循环
    for i in range(num_val):
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        # 对数据进行混洗操作
        np.random.shuffle(index_list)
        # 切分训练集和测试集
        for j in range(m):
            if j < m * 0.9:
                train_x.append(x_arr[index_list[j]])
                train_y.append(y_arr[index_list[j]])
            else:
                test_x.append(x_arr[index_list[j]])
                test_y.append(y_arr[index_list[j]])

        # 获得回归系数矩阵
        w_mat = ridge_test(train_x, train_y)
        # 循环遍历矩阵中的30组回归系数
        for k in range(30):
            # 读取训练集和数据集
            mat_test_x = np.mat(test_x)
            mat_train_x = np.mat(train_x)
            # 对数据进行标准化
            mean_train = np.mean(mat_train_x, 0)
            var_train = np.var(mat_train_x, 0)
            mat_test_x = (mat_test_x - mean_train) / var_train
            # 测试回归效果并存储
            y_est = mat_test_x * np.mat(w_mat[k, :]).T + np.mean(train_y)
            # 计算误差
            error_mat[i, k] = ((y_est.T.A - np.array(test_y)) ** 2).sum()


    # 计算误差估计值的均值
    mean_errors = np.mean(error_mat, 0)
    minMean = float(min(mean_errors))
    best_weights = w_mat[np.nonzero(mean_errors == minMean)]

    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    mean_x = np.mean(x_mat, 0)
    var_x = np.var(x_mat, 0)
    un_reg = best_weights / var_x

    # 输出构建的模型
    print("使用 Ridge 回归得到的最佳模型为:\n", un_reg)
    print("常数项为: ", -1 * sum(np.multiply(mean_x, un_reg)) + np.mean(y_mat))


if __name__ == '__main__':
    # data_mat, label_mat = load_data_set('abalone.txt')
    #
    # ridge_w = ridge_test(data_mat, label_mat)
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridge_w)
    # plt.show()

    lgX = []
    lgY = []
    set_data_collect(lgX, lgY)
    cross_validation(lgX, lgY, 10)
