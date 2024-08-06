import random
from copy import deepcopy

import numpy as np
import operator


# 载入数据
def load_file(file_path, model='normal'):
    data_mat = []
    label_mat = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if model == 'normal':
                if line[0] == 'Age':  # 跳过第一行
                    continue
                else:
                    new_line = []
                    for part in line[0:5]:  # 处理特征数据
                        if part == 'F':
                            new_line.append(1)
                        elif part == 'M':
                            new_line.append(0)
                        elif part == 'HIGH':
                            new_line.append(1)
                        elif part == 'NORMAL':
                            new_line.append(0)
                        elif part == 'LOW':
                            new_line.append(-1)
                        else:
                            new_line.append(float(part))
                    data_mat.append(new_line)
                    label_mat.append(line[-1])
            elif model == 'Tree' or model == 'tree':
                if line[0] == 'Age':  # 特征标签
                    label_mat = [feature for feature in line[:-1]]
                else:
                    new_line = []
                    for part in line[0:5]:  # 处理特征数据
                        if part == 'F':
                            new_line.append(1)
                        elif part == 'M':
                            new_line.append(0)
                        elif part == 'HIGH':
                            new_line.append(1)
                        elif part == 'NORMAL':
                            new_line.append(0)
                        elif part == 'LOW':
                            new_line.append(-1)
                        else:
                            new_line.append(round(float(part) / 10))
                    new_line.append(line[-1])
                    data_mat.append(new_line)
            else:
                raise NameError("model输入错误")
    return data_mat, label_mat


# 对数据归一化处理
def auto_norm(data_mat):
    min_val = data_mat.min(0)  # 取数据集的最小值
    max_val = data_mat.max(0)  # 取数据集的最大值
    ranges = max_val - min_val  # 确定取值的范围
    m = data_mat.shape[0]
    norm_date_set = data_mat - np.tile(min_val, (m, 1))  # 与最小值的相对值
    norm_date_set = norm_date_set / np.tile(ranges, (m, 1))  # 相对值/范围 = 归一化值
    return norm_date_set, ranges, min_val


# 规范输入向量
def modify_input(input_vec):
    if (type(input_vec[0])) == str:
        new_vec = []
        for part in input_vec:  # 处理特征数据
            if part == 'F':
                new_vec.append(1)
            elif part == 'M':
                new_vec.append(0)
            elif part == 'HIGH':
                new_vec.append(1)
            elif part == 'NORMAL':
                new_vec.append(0)
            elif part == 'LOW':
                new_vec.append(-1)
            else:
                new_vec.append(float(part))
        return new_vec
    else:
        return input_vec


#  随机插入
def random_insert(lst, item):
    index = random.randint(0, len(lst))
    lst.insert(index, item)
    return index


class Classifier:
    def trainer0(self):
        pass

    def classify0(self, input_vec):
        pass

    def test_funtion(self, test_data_path, model):
        pass


class KNN(Classifier):
    def __init__(self, data_mat, class_label, k):
        self.data_mat = np.array(data_mat)
        self.class_label = class_label
        self.k = k

    # 分类函数
    def classify0(self, input_vec):
        # 修改输入的向量
        input_vec = modify_input(input_vec)
        # 距离计算
        m, n = self.data_mat.shape
        diff_mat = np.tile(input_vec, (m, 1)) - self.data_mat  # 与数据集各行对应元素做差
        sq_diff_mat = diff_mat ** 2  # 取平方
        sq_distances = sq_diff_mat.sum(axis=1)  # 把每一行的差的平方加起来
        distance = sq_distances ** 0.5  # 取根号即最终的距离值
        sorted_dist_indices = distance.argsort()  # 对距离值排序返回原索引值
        class_count = {}
        # 选择距离最小的k点
        for i in range(self.k):
            # i访问的是当前最小的值，返回其对应的索引，即在原数据集中的行数，取得对应的标签
            vote_label = self.class_label[sorted_dist_indices[i]]
            # 统计每个标签出现的次数
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
        # iteritems讲字典变成(key, value)的元组，operator.itemgetter(1)将排序依据设置为元组的1位元素，并从大到小排
        # print(class_count)
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        # 返回出现最多的标签
        return sorted_class_count[0][0]

    """
        测试函数
        test_data_path 测试数据集路径
        ho_ratio 从数据集中用作测试的比例，默认是0.1
    """

    # 测试函数
    def test_function(self, test_data_path, ho_ratio=0.1):
        data_mat, class_labels = load_file(test_data_path)
        # norm_data_mat, ranges, min_val = auto_norm(np.array(data_mat))
        # m = norm_data_mat.shape[0]
        m = np.array(data_mat).shape[0]
        num_test_vec = int(m * ho_ratio)
        error_count = 0.0
        # 因为数据集本身是随机的，所以任取即可，这里取了前百分之10
        for i in range(num_test_vec):
            classifier_result = self.classify0(np.array(data_mat)[i, :])
            # print(f"分类结果是{classifier_result}，正确结果是{class_labels[i]}")
            # 记录错误的次数
            if classifier_result != class_labels[i]:
                error_count += 1.0
        error_rate = error_count / float(num_test_vec)
        print(f"共测试了{num_test_vec}次，总的错误率是：{round(error_rate, 5) * 100}%")
        return error_rate


class SVM(Classifier):
    def __init__(self, data_mat, class_label, c, toler, max_iter, k_tup=('rbf', 1.3)):
        # 输入数据特征矩阵
        self.data_mat = np.mat(data_mat)
        # 输入数据类别标签
        self.class_label = np.mat(class_label).T
        # 惩罚参数c
        self.c = c
        # 容忍度tolerance
        self.toler = toler
        # 样本数量
        self.m = self.data_mat.shape[0]
        # 拉格朗日乘子alphas
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # SVM模型的偏置
        self.b = 0
        # 最大迭代次数
        self.max_iter = int(max_iter)
        # 核函数相关参数
        self.k_tup = k_tup
        # 支持向量列表
        self.s_vs = np.mat([])
        self.label_sv = np.mat([])
        # 数据分组
        self.groups = self.group_data(class_label)
        # 缓存误差矩阵
        self.e_cache = np.mat(np.zeros((self.m, 2)))
        # 核矩阵
        self.k = np.mat(np.zeros((self.m, self.m)))
        # 多分类器列表
        self.multi_classifier_list = {'ovr': [], 'ovo': []}
        # 计算样本间的核函数计算结果，构成核矩阵
        for i in range(self.m):
            self.k[:, i] = self.__kernel_trans(self.data_mat, self.data_mat[i, :], k_tup)

    # 核函数
    def __kernel_trans(self, xi, xj, k_tup):
        # 读取特征矩阵的行列数
        m, n = np.shape(xi)
        # K初始化为m行1列的零向量
        K = np.mat(np.zeros((m, 1)))
        # 线性核函数只进行内积
        if k_tup[0] == 'lin':
            K = xi * xj.T
        # 高斯核函数，根据高斯核函数公式计算
        elif k_tup[0] == 'rbf':
            for j in range(m):
                delta_row = xi[j, :] - xj
                K[j] = delta_row * delta_row.T
            K = np.exp(K / (-1 * k_tup[1] ** 2))
        else:
            raise NameError('核函数无法识别')
        return K

    # 随机选择另一个α
    def __selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    # 限制α的大小
    def __clip_alpha(self, aj, h, l):
        if aj > h:
            aj = h
        if l > aj:
            aj = l
        return aj

    # 计算E
    def __calc_ek(self, k):
        f_xk = float(np.multiply(self.alphas, self.class_label).T * self.k[:, k] + self.b)
        ek = f_xk - float(self.class_label[k])
        return ek

    # 用于选择第二个α值，返回索引和误差值
    def __selectJ(self, i, ei):
        max_k = -1
        max_delta_e = 0
        ej = 0
        self.e_cache[i] = [1, ei]
        valid_ecache_list = np.nonzero(self.e_cache[:, 0].A)[0]  # 获取有效值的行索引
        # 从现有的有效误差中寻找第二个alpha
        if len(valid_ecache_list) > 1:
            # 遍历，找到最大的Ek
            for k in valid_ecache_list:
                if k == i:
                    continue
                ek = self.__calc_ek(k)
                delta_e = abs(ei - ek)
                if delta_e > max_delta_e:
                    max_k = k
                    max_delta_e = delta_e
                    ej = ek
            return max_k, ej
        else:
            # 如果现有有效误差只有1个（即第一个α），则说明这是第一次循环，直接在所有数据里面进行随机选择
            j = self.__selectJrand(i, self.m)
            ej = self.__calc_ek(j)
        return j, ej

        # 计算误差值，并存入缓存

    # 更新误差值
    def __update_ek(self, k):
        ek = self.__calc_ek(k)
        self.e_cache[k] = [1, ek]

    # 外循环函数
    # 控制SMO算法的迭代流程，更新alpha、b的值
    def __smo(self):
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0
        # 遍历整个数据集alpha都没有更新或者超过最大迭代次数，则退出循环
        while (iter < self.max_iter) and ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            # 遍历所有的值
            if entire_set:
                for i in range(self.m):
                    alpha_pairs_changed += self.__inner(i)
                    # print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
                iter += 1
            # 遍历非边界值
            else:
                # 取出非边界值的行索引
                non_bound_i_s = np.nonzero(np.array((self.alphas.A > 0) * (self.alphas.A < self.c)))[0]
                for i in non_bound_i_s:
                    alpha_pairs_changed += self.__inner(i)
                    # print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
                iter += 1
            # 如果上次循环是遍历所有值，那么下次循环改为遍历非边界值
            if entire_set:
                entire_set = False
            # 如果上次循环是遍历非边界值但没有收获，则改为遍历所有值
            elif alpha_pairs_changed == 0:
                entire_set = True
            # print("迭代次数:%d" % iter)
        return self.b, self.alphas

        # 内循环函数

    # 内循环函数
    # 具体实现对alpha、b的更新
    def __inner(self, i):
        ei = self.__calc_ek(i)
        if ((self.class_label[i] * ei < -self.toler) and (self.alphas[i] < self.c)) or \
                ((self.class_label[i] * ei > self.toler) and (self.alphas[i] > 0)):
            j, ej = self.__selectJ(i, ei)
            alpha_i = self.alphas[i].copy()
            alpha_j = self.alphas[j].copy()
            if self.class_label[i] != self.class_label[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.c, self.c + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[i] + self.alphas[j] - self.c)
                H = min(self.c, self.alphas[i] + self.alphas[j])
            if L == H:
                # print("L == H")
                return 0
            # eta是alpha[j]的最优修改量
            eta = (2.0 * self.data_mat[i, :] * self.data_mat[j, :].T - self.data_mat[i, :] * self.data_mat[i, :].T -
                   self.data_mat[j, :] * self.data_mat[j, :].T)
            # eta >= 0的情况比较少，并且优化过程计算复杂，所以此处做了简化处理，直接跳过了
            if eta >= 0:
                # print("eta >= 0")
                return 0
            self.alphas[j] -= self.class_label[j] * (ei - ej) / eta
            self.alphas[j] = self.__clip_alpha(self.alphas[j], H, L)
            self.__update_ek(j)
            # 比对原值，看变化是否明显，如果优化并不明显则退出
            if abs(self.alphas[j] - alpha_j) < 0.00001:
                # print("j not moving enough")
                return 0
            self.alphas[i] += self.class_label[j] * self.class_label[i] * (alpha_j - self.alphas[j])
            self.__update_ek(i)
            b1 = self.b - ei - self.class_label[i] * (self.alphas[i] - alpha_i) * self.data_mat[i, :] * self.data_mat[i,
                                                                                                        :].T - \
                 self.class_label[j] * (self.alphas[j] - alpha_j) * self.data_mat[i, :] * self.data_mat[j, :].T
            b2 = self.b - ej - self.class_label[i] * (self.alphas[i] - alpha_i) * self.data_mat[i, :] * self.data_mat[j,
                                                                                                        :].T - \
                 self.class_label[j] * (self.alphas[j] - alpha_j) * self.data_mat[j, :] * self.data_mat[j, :].T
            if (0 < self.alphas[i]) and (self.c > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.c > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    # 根据类标签给数据分组
    def group_data(self, class_label):
        data_group = {}
        for i in range(self.m):
            label = class_label[i]
            if label not in data_group:
                data_group[label] = 1
            else:
                data_group[label] += 1
        self.groups = data_group
        return data_group

    # 训练器
    def trainer0(self):
        # 优化alpha和b
        self.__smo()
        # 获得支持向量列表
        sv_ind = np.nonzero(self.alphas.A > 0)[0]
        self.s_vs = self.data_mat[sv_ind]
        self.label_sv = self.class_label[sv_ind]

    # def trainer1(self, data_mat, class_labels):

    # 分类器
    def classify0(self, input_vec):
        # 修改输入的向量
        input_vec = modify_input(input_vec)
        # 计算各个点的核
        kernel_eval = self.__kernel_trans(self.s_vs, np.mat(input_vec), self.k_tup)
        # 根据支持向量的点计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(self.label_sv, self.alphas[np.nonzero(self.alphas.A > 0)[0]]) + self.b
        predict = np.sign(predict)[0]
        return predict

    # 采用一对多策略的多分类器
    def ovr_trainer0(self, max_iter=3):
        # 对每个类别
        for iter in self.groups:
            label_mat = []
            # 与当前类别相同标签为1，其他为-1
            for i in range(self.m):
                if self.class_label[i, 0] == iter:  # 类别相同
                    label_mat.append(1)
                else:  # 类别不同
                    label_mat.append(-1)

            # 用新的标签列表label_arr初始化一个SVM对象
            svm_multi_classifier = SVM(self.data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)

            # 最小训练错误率
            min_error = 1.0
            # 保存训练错误率最小的一次
            best_classier = deepcopy(svm_multi_classifier)

            for num_iter in range(max_iter):
                svm_multi_classifier.trainer0()
                # 错误率
                error_count = 0.0
                # 当前训练下的预测错误率
                for i in range(svm_multi_classifier.m):
                    predict = svm_multi_classifier.classify0(svm_multi_classifier.data_mat[i, :])
                    if np.sign(predict) != np.sign(svm_multi_classifier.class_label[i]):
                        error_count += 1
                this_rate = error_count / float(svm_multi_classifier.m)
                # 如果错误率小于最小训练错误率,更新配置
                if this_rate < min_error:
                    min_error = this_rate
                    best_classier = deepcopy(svm_multi_classifier)

            self.multi_classifier_list['ovr'].append(best_classier)
            print(f"对于{iter}最小的训练错误率是{round(min_error * 100, 3)}%")
        return self.multi_classifier_list['ovr']

    def ovr_trainer1(self):
        # 对每个类别
        for iter in self.groups:
            remain_data_index = []
            data_count = {label: 1 for label in self.groups}
            data_mat = []
            label_mat = []
            if self.groups[iter] < int(self.m / 4):
                sample_num = int(self.m / 4 / (len(self.groups) - 1))
                # 与当前类别相同标签为1，其他为-1
                for i in range(self.m):
                    if self.class_label[i, 0] == iter:  # 类别相同
                        label_mat.append(1)
                        if len(data_mat) == 0:  # 加入第一个样本时
                            data_mat = np.vstack(self.data_mat[i])
                        else:
                            data_mat = np.vstack([data_mat, self.data_mat[i]])

                    else:  # 类别不同
                        if len(data_mat) == 0:
                            label_mat.append(-1)
                            data_mat = np.vstack(self.data_mat[i])

                        else:
                            if data_count[self.class_label[i, 0]] <= sample_num:
                                label_mat.append(-1)
                                data_count[self.class_label[i, 0]] += 1
                                data_mat = np.vstack([data_mat, self.data_mat[i]])

                            else:
                                remain_data_index.append(i)

                # 取1/3的剩余样本
                num_remain_data = len(remain_data_index)
                part_remain_sample_index = np.random.choice(remain_data_index, int(num_remain_data / 3), replace=False)
                data_mat = np.vstack([data_mat, self.data_mat[part_remain_sample_index]])
                label_mat.extend(-1 for i in range(len(part_remain_sample_index)))

            else:
                # 与当前类别相同标签为1，其他为-1
                for i in range(self.m):
                    if self.class_label[i, 0] == iter:  # 类别相同
                        label_mat.append(1)
                    else:  # 类别不同
                        label_mat.append(-1)
                data_mat = self.data_mat

            # 用新的标签列表label_arr初始化一个SVM对象
            svm_multi_classifier = SVM(data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)

            # 最小训练错误率
            min_error = 1.0
            # 保存训练错误率最小的一次
            best_classier = deepcopy(svm_multi_classifier)

            for num_iter in range(3):
                svm_multi_classifier.trainer0()
                # 错误率
                error_count = 0.0
                # 当前训练下的预测错误率
                for i in range(svm_multi_classifier.m):
                    predict = svm_multi_classifier.classify0(svm_multi_classifier.data_mat[i, :])
                    if np.sign(predict) != np.sign(svm_multi_classifier.class_label[i]):
                        error_count += 1
                this_rate = error_count / float(svm_multi_classifier.m)
                # 如果错误率小于最小训练错误率,更新配置
                if this_rate < min_error:
                    min_error = this_rate
                    best_classier = deepcopy(svm_multi_classifier)

            self.multi_classifier_list['ovr'].append(best_classier)
            print(f"对于{iter}最小的训练错误率是{round(min_error * 100, 3)}%")
        return self.multi_classifier_list['ovr']

    # 采用一对一策略的分类器
    def ovo_trainer0(self):
        import itertools
        # 获取所有可能的类别组合
        class_combinations = list(itertools.combinations(self.groups.keys(), 2))

        # 对每一个组合
        for class_pair in class_combinations:
            data_mat = []
            label_mat = []
            # 选择当前一对类别的样本索引
            for i in range(self.m):
                if self.class_label[i, 0] == class_pair[0]:  # 正类
                    label_mat.append(1)
                    if len(data_mat) == 0:  # 加入第一个样本时
                        data_mat = np.vstack(self.data_mat[i])
                    else:
                        data_mat = np.vstack([data_mat, self.data_mat[i]])
                elif self.class_label[i, 0] == class_pair[1]:  # 负类
                    label_mat.append(-1)
                    if len(data_mat) == 0:  # 加入第一个样本时
                        data_mat = np.vstack(self.data_mat[i])
                    else:
                        data_mat = np.vstack([data_mat, self.data_mat[i]])

            # 创建一个新的SVM
            svm_multi_classifier = SVM(data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)

            # 最小训练错误率
            min_error = 1.0
            # 保存训练错误率最小的一次
            best_classier = deepcopy(svm_multi_classifier)

            for num_iter in range(3):
                svm_multi_classifier.trainer0()
                # 错误率
                error_count = 0.0
                # 当前训练下的预测错误率
                for i in range(svm_multi_classifier.m):
                    predict = svm_multi_classifier.classify0(svm_multi_classifier.data_mat[i, :])
                    if np.sign(predict) != np.sign(svm_multi_classifier.class_label[i]):
                        error_count += 1
                this_rate = error_count / float(svm_multi_classifier.m)
                # 如果错误率小于最小训练错误率,更新配置
                if this_rate < min_error:
                    min_error = this_rate
                    best_classier = deepcopy(svm_multi_classifier)

            self.multi_classifier_list['ovr'].append(best_classier)
            # print(f"对于{class_pair}最小的训练错误率是{round(min_error * 100, 3)}%")
        return self.multi_classifier_list['ovr']

    def ovr_classify0(self, input_vec):
        group_list = list(self.groups.keys())
        result_list = []
        group_index = 0
        for classifier in self.multi_classifier_list['ovr']:
            predict = classifier.classify0(input_vec)
            result_list.append([group_list[group_index], predict])
            group_index += 1
        # 取最大的值作为分类结果，也就是正的越大越可信
        result_list.sort(key=lambda x: x[1], reverse=True)
        classify_result = result_list[0][0]
        return classify_result

    def ovo_classify0(self, input_vec):
        # 保存投票结果
        count_label = {label: 0 for label in self.groups}

        for class_pair in self.multi_classifier_list['ovo']:
            classifier = class_pair[1]
            predict = classifier.classify0(input_vec)
            if np.sign(predict) > 0:  # 分类结果为正类
                count_label[class_pair[0][0]] += 1  # 次数加1
            else:  # 分类接过为负类
                count_label[class_pair[0][1]] += 1  # 次数加1

        # 按照类别在训练样本中出现的频率作为权重排序，避免多个类别投票数量一致
        sorted_count_label = tuple(sorted(count_label.items(), key=lambda x: (self.groups[x[0]]/self.m) * x[1], reverse=True))
        # 升序排序，取第一个作为分类结果
        class_result = sorted_count_label[0][0]
        return class_result

    # 测试器
    def test_function(self, test_data_path, ratio=0.1, mode='bin'):
        data_mat, class_labels = load_file(test_data_path)
        data_mat = np.mat(data_mat)
        m = data_mat.shape[0]
        num_test_vec = int(m * ratio)
        error_count = 0
        error_rate = 0.0
        if mode == 'bin':
            # print(f"这里有{self.s_vs.shape[0]}支持向量")
            # 因为数据集本身是随机的，所以任取即可，这里取了前百分之10
            for i in range(num_test_vec):
                predict = self.classify0(data_mat[i, :])
                if np.sign(predict) != np.sign(class_labels[i]):
                    error_count += 1
            error_rate = round(error_count / float(num_test_vec), 5)
            print(f"共测试了{num_test_vec}次，总的错误率是：{error_rate * 100}%")

        elif mode == 'ovr':
            for i in range(num_test_vec):
                classify_result = self.ovr_classify0(data_mat[i, :])
                if classify_result != class_labels[i]:
                    error_count += 1
                error_rate = round(error_count / float(num_test_vec), 5)
            print(f"共测试了{num_test_vec}次，总的错误率是：{error_rate * 100}%")

        elif mode == 'ovo':
            for i in range(num_test_vec):
                classify_result = self.ovo_classify0(data_mat[i, :])
                if classify_result != class_labels[i]:
                    error_count += 1
                error_rate = round(error_count / float(num_test_vec), 5)

            print(f"共测试了{num_test_vec}次，总的错误率是：{error_rate * 100}%")
        else:
            raise NameError("测试模式错误，'bin'为二分类，'ovr'、'ovo'为多分类")

        return error_rate


"""
    决策树类及其需要的函数
"""


# 计算香农熵
def calc_shannon_ent(dataSet):
    from math import log
    num_entries = len(dataSet)
    label_count = {}
    # 计算每个标签出现的概率
    for feat_vec in dataSet:
        current_label = feat_vec[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    # 计算香农熵
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# 按照给定特征划分数据集
def split_data_set(data_mat, feature, value):
    return_data_mat = []
    for feat_vec in data_mat:
        # 对指定的特征值进行筛选
        if feat_vec[feature] == value:
            # 如果特征值符合，将样本中这个特征剔除后的新向量归类到新的列表里
            reduced_feat_vec: list = feat_vec[:feature]
            reduced_feat_vec.extend(feat_vec[feature + 1:])
            return_data_mat.append(reduced_feat_vec)
    return return_data_mat


# 得到分类后可以获得最大信息熵的分类特征
def choose_best_feature(data_mat):
    num_feat = len(data_mat[0]) - 1
    base_entropy = calc_shannon_ent(data_mat)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(num_feat):
        # 将每个样本的第i个特征加入列表
        feat_list = [example[i] for example in data_mat]
        # 唯一特征值
        unique_val = set(feat_list)
        # 初始的香农熵
        new_entropy = 0.0
        # 求条件熵并求和
        for val in unique_val:
            sub_data_set = split_data_set(data_mat, i, val)
            prob = len(sub_data_set) / float(len(data_mat))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 获得信息增益，信息增益大的即为最好特征规划
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


# 筛选出出现次数最多的标签
def majorityCnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建决策树
def createTree(data_mat, feature_label):
    # 存放每个样本的类标签
    class_list = [example[-1] for example in data_mat]
    if class_list.count(class_list[0]) == len(class_list):  # class_list内的类标签完全相同时，停止继续划分
        return class_list[0]
    if len(data_mat[0]) == 1:  # 使用完了所有的特征，仍不能将数据集划分为包含唯一类别的分组
        return majorityCnt(class_list)  # 此时使用出现最多的类别作为返回值
    best_feat = choose_best_feature(data_mat)
    best_feat_label = feature_label[best_feat]
    my_tree = {best_feat_label: {}}
    # del (labels[best_feat])
    new_labels = []
    new_labels.extend(feature_label)
    del (new_labels[best_feat])
    # 取每个样本在最好特征处的特征值
    feat_val = [example[best_feat] for example in data_mat]
    unique_val = set(feat_val)
    # 创建当前最好特征处各特征值对应的分支
    for val in unique_val:
        sub_labels = new_labels[:]
        # print(f'{best_feat_label}, {val}')
        my_tree[best_feat_label][val] = createTree(split_data_set(data_mat, best_feat, val), sub_labels)
    return my_tree


# 从保存决策树的文件获得对应的树
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def tree_classify(input_tree, featLabel, testVec):
    class_label = ''
    first_str = list(input_tree.keys())[0]
    second_dice = input_tree[first_str]
    # 用index()返回分类特征对应索引
    feat_index = featLabel.index(first_str)
    for key in second_dice.keys():
        if testVec[feat_index] == key:
            if type(testVec[feat_index]).__name__ == 'dict':
                class_label = tree_classify(second_dice[key], featLabel, testVec)
            else:
                class_label = second_dice[key]
    return class_label


class Tree(Classifier):
    def __init__(self, data_mat, feature_label):
        self.data_mat = data_mat
        self.feature_label = feature_label
        self.tree = {}

    # 创建树
    def trainer0(self):
        self.tree = createTree(self.data_mat, self.feature_label)
        return self.tree

    # 保存树
    def storeTree(self, filename):
        import pickle
        fw = open(filename, 'wb')
        pickle.dump(self.tree, fw)
        fw.close()

    # 测试算法
    def classify0(self, input_vec):
        # 修改输入的向量
        input_vec = modify_input(input_vec)
        # 调用递归函数classify进行分类
        classify_result = tree_classify(self.tree, self.feature_label, input_vec)
        return classify_result

    def test_function(self, test_data_path, ho_ratio=0.1, mode='mul'):
        test_data_mat, feature_labels = load_file(test_data_path, 'Tree')
        # 如果测试数据集没有特征行
        if not feature_labels:
            feature_labels = self.feature_label
        data_mat = [line[:-1] for line in test_data_mat]
        class_labels = [line[-1] for line in test_data_mat]
        m = len(data_mat)
        num_test_vec = int(m * ho_ratio)
        error_count = 0
        for i in range(num_test_vec):
            predict = self.classify0(data_mat[i])
            print((i, predict))
            if predict != class_labels[i]:
                error_count += 1
        error_rate = round(error_count / float(num_test_vec), 5)
        print(f"共测试了{num_test_vec}次，总的错误率是：{error_rate * 100}%")
        return error_rate


def loadDataSet(filename):
    data_mat = []
    label_mat = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip().split('\t')
        data_mat.append([float(line[0]), float(line[1])])
        label_mat.append(int(float(line[-1])))
    return data_mat, label_mat


if __name__ == '__main__':
    # data, labels = load_file('./data/drug.txt')
    # num_data = np.mat(data).shape[0]
    # ho_ratio = 0.30  # 取数据集的百分之13用作测试
    # num_test_iter = int(ho_ratio * num_data)
    # # 创建SVM分类器对象
    # svm_classifier = SVM(data[num_test_iter: num_data],
    #                      labels[num_test_iter:num_data], c=150, toler=0.0001, max_iter=100,
    #                      k_tup=('rbf', 1.3))
    # # svm_classifier.ovr_trainer1()
    # svm_classifier.ovo_trainer0()
    # svm_classifier.test_function('./data/drug.txt', ho_ratio, 'ovo')
    #
    # num_data = np.mat(data).shape[0]
    # ho_ratio = 0.14
    # num_test_iter = int(ho_ratio * num_data)
    #
    # svm_classifier2 = SVM(data[num_test_iter: num_data],
    #                       labels[num_test_iter:num_data], 300, 0.0001, 300, ('rbf', 5))
    # svm_classifier2.ovr_trainer0()
    # err_rate = svm_classifier2.test_function('./data/drug.txt', ho_ratio, 'ovo')

    data, labels = load_file('../classify_medication/data/drug.txt')
    num_data = np.mat(data).shape[0]
    min_error_rate = 1
    best_k = 0.5
    best_c = 100
    best_test_size = 0.1

    for test_size in np.arange(0.1, 0.31, 0.01):
        test_size = round(test_size, 2)
        num_test_iter = int(test_size * num_data)
        svm_classifier2 = SVM(data[num_test_iter: num_data],
                              labels[num_test_iter:num_data], best_c, 0.0001, 100, k_tup=('rbf', best_k))
        svm_classifier2.ovr_trainer1()
        print(f"k={best_k}, c={best_c}, num_test_sample={num_test_iter}")
        this_rate = svm_classifier2.test_function('../classify_medication/data/drug.txt', test_size, 'ovr')
        if min_error_rate > this_rate:
            min_error_rate = this_rate
            best_test_size = test_size
            if min_error_rate < 0.15:
                break

    for c in range(100, 210, 10):
        num_test_iter = int(best_test_size * num_data)
        svm_classifier2 = SVM(data[num_test_iter: num_data],
                              labels[num_test_iter:num_data], c, 0.0001, 100, k_tup=('rbf', best_k))
        svm_classifier2.ovr_trainer1()
        print(f"k={best_k}, c={c}, num_test_sample={num_test_iter}")
        this_rate = svm_classifier2.test_function('../classify_medication/data/drug.txt', best_test_size, 'ovr')
        if min_error_rate > this_rate:
            min_error_rate = this_rate
            best_c = c
            if min_error_rate < 0.15:
                break

    for k in np.arange(0.5, 2, 0.1):
        k = round(k, 1)
        num_test_iter = int(best_test_size * num_data)
        svm_classifier2 = SVM(data[num_test_iter: num_data],
                              labels[num_test_iter:num_data], best_c, 0.0001, 100, k_tup=('rbf', k))
        svm_classifier2.ovr_trainer1()
        print(f"k={k}, c={best_c}, num_test_sample={num_test_iter}")
        this_rate = svm_classifier2.test_function('../classify_medication/data/drug.txt', best_test_size, 'ovr')
        if min_error_rate > this_rate:
            min_error_rate = this_rate
            best_k = k
            if min_error_rate < 0.15:
                break

    print(f"最小的测试错误率：{min_error_rate},c={best_c}, k={best_k}, rate={best_test_size}")

    # import multiprocessing
    # import woke_module
    # import numpy as np
    #
    # # Assuming you have 'load_file', 'SVM' class, and 'num_data' defined somewhere
    # data, labels = load_file('./data/drug.txt')
    # min_error_rate = 1
    # best_k = 1
    # best_c = 1
    # best_test_rate = 0.0
    #
    #
    #
    #
    # if __name__ == '__main__':
    #     num_data = np.mat(data).shape[0]
    #     params_list = [(round(k, 1), c, num_data, data, labels) for k in np.arange(0.5, 2, 0.1) for c in range(100, 210, 10)]
    #
    #     pool = multiprocessing.Pool()
    #     results = pool.map(woke_module.worker, params_list)
    #     pool.close()
    #     pool.join()
    #
    #     for result in results:
    #         if result[0] < min_error_rate:
    #             min_error_rate, best_c, best_k, best_test_rate = result
    #
    #     print(f"最小的测试错误率：{min_error_rate}, c={best_c}, k={best_k}, rate={best_test_rate}")
    def ovr_trainer1(self):
        # 对每个类别
        for iter in self.groups:
            remain_data_index = []
            data_count = {label: 1 for label in self.groups}
            data_mat = []
            label_mat = []
            # 当前类别的样本数量较少
            if self.groups[iter] < int(self.m / 4):
                sample_num = int(self.m / 4 / (len(self.groups) - 1))
                # 与当前类别相同标签为1，其他为-1
                for i in range(self.m):
                    if self.class_label[i, 0] == iter:  # 类别相同
                        label_mat.append(1)
                        if len(data_mat) == 0:  # 加入第一个样本时
                            data_mat = np.vstack(self.data_mat[i])
                        else:
                            data_mat = np.vstack([data_mat, self.data_mat[i]])

                    else:  # 类别不同
                        if len(data_mat) == 0:
                            label_mat.append(-1)
                            data_mat = np.vstack(self.data_mat[i])

                        else:
                            if data_count[self.class_label[i, 0]] <= sample_num:
                                label_mat.append(-1)
                                data_count[self.class_label[i, 0]] += 1
                                data_mat = np.vstack([data_mat, self.data_mat[i]])

                            else:
                                remain_data_index.append(i)

                # 取1/3的剩余样本
                num_remain_data = len(remain_data_index)
                part_remain_sample_index = np.random.choice(remain_data_index, int(num_remain_data / 3), replace=False)
                data_mat = np.vstack([data_mat, self.data_mat[part_remain_sample_index]])
                label_mat.extend(-1 for i in range(len(part_remain_sample_index)))

            else:
                # 与当前类别相同标签为1，其他为-1
                for i in range(self.m):
                    if self.class_label[i, 0] == iter:  # 类别相同
                        label_mat.append(1)
                    else:  # 类别不同
                        label_mat.append(-1)
                data_mat = self.data_mat

            # 用新的标签列表label_arr初始化一个SVM对象
            svm_multi_classifier = SVM(data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)
            svm_multi_classifier.trainer0()

            self.multi_classifier_list['ovr'].append(svm_multi_classifier)

        return self.multi_classifier_list['ovr']