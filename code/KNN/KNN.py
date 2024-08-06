from os import listdir

import numpy as np
import operator

from past.builtins import raw_input


def createDateSet():
    group = np.array([[1.0, 1.1], [1.0, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX: np.array, dataSet: np.array, labels: list, k: int):
    # 距离计算
    data_set_size = dataSet.shape[0]  # 获得数据集行数
    diff_mat = np.tile(inX, (data_set_size, 1)) - dataSet  # 与数据集各行对应元素做差
    sq_diff_mat = diff_mat ** 2  # 取平方
    sq_distances = sq_diff_mat.sum(axis=1)  # 把每一行的差的平方加起来
    distance = sq_distances ** 0.5  # 取根号即最终的距离值
    sorted_dist_indices = distance.argsort()  # 对距离值排序返回原索引值
    class_count = {}
    # 选择距离最小的k点
    for i in range(k):
        # i访问的是当前最小的值，返回其对应的索引，即在原数据集中的行数，取得对应的标签
        vote_label = labels[sorted_dist_indices[i]]
        # 统计每个标签出现的次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # iteritems讲字典变成(key, value)的元组，operator.itemgetter(1)将排序依据设置为元组的1位元素，并从大到小排
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现最多的标签
    return sorted_class_count[0][0]


def file2mat(filename):
    f = open(filename, 'r', encoding="UTF-8")
    array_lines = f.readlines()
    count_lines = len(array_lines)
    date_mat = np.zeros((count_lines, 3))
    class_label = []
    f.close()
    i = 0
    for line in array_lines:
        line = line.strip()
        new_line = line.split('\t')
        date_mat[i, :] = new_line[0:3]
        if "didntLike" == new_line[-1]:
            class_label.append(1)
        if "smallDoses" == new_line[-1]:
            class_label.append(2)
        if "largeDoses" == new_line[-1]:
            class_label.append(3)
        i += 1
    return date_mat, class_label


def file2person(filename):
    f = open(filename, 'r', encoding="UTF-8")
    array_lines = f.readlines()
    count_lines = len(array_lines)
    person_mat = np.zeros((count_lines, 3))
    i = 0
    person_name = []
    for line in array_lines:
        line = line.strip()
        new_line = line.split(',')
        person_mat[i, :] = new_line[0:3]
        person_name.append(new_line[-1])
        i += 1
    return person_mat, person_name


def img2vector(filename):
    return_vec = np.zeros((1, 1024))
    f = open(filename, 'r', encoding='UTF-8')
    # 将32×32的二进制图像矩阵转换成1×1024的向量
    # readline()每次读一行，每次读32个二进制数，一共读32次
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            return_vec[0, 32*i+j] = int(line_str[j])
    return return_vec


# 归一化
def autoNorm(data_set):
    min_val = data_set.min(0)  # 取数据集的最小值
    max_val = data_set.max(0)  # 取数据集的最大值
    ranges = max_val - min_val  # 确定取值的范围
    norm_date_set = np.zeros_like(data_set)  # 准备一个用于存放归一化数据的矩阵
    m = data_set.shape[0]
    norm_date_set = data_set - np.tile(min_val, (m, 1))  # 与最小值的相对值
    norm_date_set = norm_date_set / np.tile(ranges, (m, 1))  # 相对值/范围 = 归一化值
    print(type(data_set))
    return norm_date_set, ranges, min_val


def datingClassTest():
    ho_ratio = 0.10  # 取百分之十的数据作为测试数据
    dating_data_mat, dating_labels = file2mat("./datingTestSet.txt")
    norm_data_mat, ranges, min_val = autoNorm(dating_data_mat)
    m = norm_data_mat.shape[0]
    num_test_vec = int(m * ho_ratio)
    error_count = 0.0
    # 因为数据集本身是随机的，所以任取百分之十即可，这里取了前百分之10
    for i in range(num_test_vec):
        classifier_result = classify0(norm_data_mat[i, :],
                                      norm_data_mat[num_test_vec:m, :],
                                      dating_labels[num_test_vec:m], k=5)
        # print(f"分类结果是{classifier_result}，正确结果是{dating_labels[i]}")
        # 记录错误的次数
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(f"共测试了{num_test_vec}次，总的错误率是：{error_count / float(num_test_vec)}")


def handwritingClassTest():
    hw_labels = []
    # 通过os导入listdir获得目录名
    training_file_list = listdir('./digits/trainingDigits')
    m_training = len(training_file_list)
    training_mat = np.zeros((m_training, 1024))
    # 准备训练数据集
    for i in range(m_training):
        # 通过解析文件名获得标签
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_label_str = int(file_str.split('_')[0])
        hw_labels.append(class_label_str)
        training_mat[i, :] = img2vector(f'./digits/trainingDigits/{file_name_str}')
    # 准备测试数据集
    test_file_list = listdir('./digits/testDigits')
    m_test = len(test_file_list)
    error = 0
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_label_str = int(file_str.split('_')[0])
        test_vec = img2vector(f'./digits/testDigits/{file_name_str}')

        classifier_result = classify0(test_vec, training_mat, hw_labels, 5)
        print(f"预测分类是：{classifier_result}，正确结果是{class_label_str}")
        if classifier_result != class_label_str:
            error += 1
    error_rate = "{:.4f}%".format(error/float(m_test)*100)
    print(f"总共测试了{m_test}组数据\n最终的错误率是{error_rate}")


def classifyPerson(person_data, person_name):
    result_list = ['不喜欢的', '一般有魅力的', '很有魅力的']
    love_person_list = []
    # ff_miles = float(raw_input("每年获得的飞行常客里程数"))
    # percent_tats = float(raw_input("玩视频游戏占时间的百分比"))
    # ice_cream = float(raw_input("每周消耗冰淇淋的公升数"))

    # ff_miles = 34500
    # percent_tats = 13
    # ice_cream = 8

    dating_mat, dating_labels = file2mat('./datingTestSet.txt')
    norm_mat, ranges, min_val = autoNorm(dating_mat)
    i = 0
    person_num = len(person_name)
    for person in person_data:
        in_arr = np.array(person)
        classify_result = classify0((in_arr - min_val) / ranges, norm_mat, dating_labels, 4)
        # 　print(f"你对这个人的感觉是：{result_list[classify_result-1]}")
        if '一般有魅力的' == result_list[classify_result - 1] or '很有魅力的' == result_list[classify_result - 1]:
            love_person_list.append(person_name[i])
        i += 1
    print(f"附近有{person_num}个人")
    print(f"你可能感兴趣的人是：{love_person_list}")
