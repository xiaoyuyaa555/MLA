from math import log
import operator


def file2lenses(filename):
    f = open(filename, 'r')
    data = f.readlines()
    data_list = []
    for line in data:
        line = line.strip()
        data_list.append(line.split('\t'))
    f.close()
    return data_list


def createDateSet():
    data_set = [[1, 1, 'yes'], [1, 0, 'no'], [1, 1, 'yes'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


# 计算香农熵
def calcShannonEnt(dataSet):
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
def splitDataSet(data_set, axis, value):
    return_data_set = []
    for feat_vec in data_set:
        # 对指定的特征值进行筛选
        if feat_vec[axis] == value:
            # 如果特征值符合，将样本中这个特征剔除后的新向量归类到新的列表里
            reduced_feat_vec: list = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            return_data_set.append(reduced_feat_vec)
    return return_data_set


def chooseBestFeatureToSplit(data_set):
    num_feat = len(data_set[0]) - 1
    base_entropy = calcShannonEnt(data_set)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(num_feat):
        # 将每个样本的第i个特征加入列表
        feat_list = [example[i] for example in data_set]
        # 唯一特征值
        unique_val = set(feat_list)
        # 初始的香农熵
        new_entropy = 0.0
        # 求条件熵并求和
        for val in unique_val:
            sub_data_set = splitDataSet(data_set, i, val)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calcShannonEnt(sub_data_set)
        # 获得信息增益，信息增益大的即为最好特征规划
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


# 筛选出出现次数最多的标签，类似投票表决
def majorityCnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建树
def createTree(data_set, labels):
    # 存放每个样本的类标签
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # class_list内的类标签完全相同时，停止继续划分
        return class_list[0]
    if len(data_set[0]) == 1:                               # 使用完了所有的特征，仍不能将数据集划分为包含唯一类别的分组
        return majorityCnt(class_list)                      # 此时使用出现最多的类别作为返回值
    best_feat = chooseBestFeatureToSplit(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    # del (labels[best_feat])
    new_labels = []
    new_labels.extend(labels)
    del (new_labels[best_feat])
    # 取每个样本在最好特征处的特征值
    feat_val = [example[best_feat] for example in data_set]
    unique_val = set(feat_val)
    # 创建当前最好特征处各特征值对应的分支
    for val in unique_val:
        sub_labels = new_labels[:]
        # print(f'{best_feat_label}, {val}')
        my_tree[best_feat_label][val] = createTree(splitDataSet(data_set, best_feat, val), sub_labels)
    return my_tree


def storeTree(input_tree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# 测试算法
def classify(input_tree, feat_labels, test_vec):
    class_label = ''
    firstStr = list(input_tree.keys())[0]
    second_dict = input_tree[firstStr]
    featIndex = feat_labels.index(firstStr)  # index方法查找当前列表中第一个匹配firstStr变量的元素的索引
    for key in second_dict.keys():
        if test_vec[featIndex] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label
