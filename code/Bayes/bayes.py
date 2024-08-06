from numpy import *
import re



def loadDataSet():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return posting_list, class_vec


# 利用set创建一个包含所有不重复词的列表
def createVocabList(dataSet):
    vocab_set = set([])
    for document in dataSet:
        vocab_set = vocab_set | set(document)  # 取两个集合的并集
    return list(vocab_set)


# 检查输入文档种的单词是否在词汇表内
# 这是贝努利模型也称词集模型，不考虑出现次数，只考虑是否出现，即认为假设词是等权重的
def setOfWords2Vec(vocab_list, inputSet):
    return_vec = [0] * len(vocab_list)
    for word in inputSet:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print(f"the word:{word} is not in my vocabulary!")
    return return_vec


# 这是多项式模型也称词袋模型，考虑出现的次数，关心出现的频率
def bagOfWords2Vec(vocab_list, inputSet):
    return_vec = [0] * len(vocab_list)
    for word in inputSet:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print(f"the word:{word} is not in my vocabulary!")
    return return_vec


def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    # 认为每一篇文档的各个特征都是一样的
    num_words = len(train_matrix[0])
    # 计算含侮辱性词汇的概率
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:  # 含侮辱性词汇
            p1_num += train_matrix[i]  # 将侮辱性词汇个数+1
            p1_denom += sum(train_matrix[i])  # class1总词汇数增加
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p0_vec = log(p0_num / p0_denom)  # 对应词汇个数/总词汇数
    p1_vec = log(p1_num / p1_denom)
    return p0_vec, p1_vec, p_abusive


def classifyNB(input_vec, p0_vec, p1_vec, p_class1):
    p1 = sum(input_vec * p1_vec) + log(p_class1)
    p0 = sum(input_vec * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    list_of_posts, list_of_classes = loadDataSet()
    my_vocab_list = createVocabList(list_of_posts)
    train_mat = []
    for post_in_doc in list_of_posts:
        train_mat.append(setOfWords2Vec(my_vocab_list, post_in_doc))
    p0_vec, p1_vec, p_abusive = trainNB0(array(train_mat), array(list_of_classes))
    test_post1 = ['love', 'my', 'dalmation']
    test_doc = array(setOfWords2Vec(my_vocab_list, test_post1))
    print(f"{test_post1} classified as: {classifyNB(test_doc, p0_vec, p1_vec, p_abusive)}")
    test_post1 = ['stupid', 'garbage', 'love', 'my']
    test_doc = array(setOfWords2Vec(my_vocab_list, test_post1))
    print(f"{test_post1} classified as: {classifyNB(test_doc, p0_vec, p1_vec, p_abusive)}")


def getFileData():
    f = open('./email.txt', 'r', encoding='UTF-8')
    class_vec = []
    doc_vec = []
    for email_lines in f:
        # 先按照制表符划分成标签和文本内容两部分
        email_lines = email_lines.split('\t')
        # 根据标签生成类别列表
        if email_lines[0] == 'ham':
            class_vec.append(0)
        elif email_lines[0] == 'spam':
            class_vec.append(1)
        # 通过正则表达式去除除字母和数字的字符串
        email_doc = re.split(r'[\W*]', email_lines[1])
        # 筛选长度大于2的单词，并将其变为小写
        this_doc = [tok.lower() for tok in email_doc if len(tok) > 2]
        doc_vec.append(this_doc)
    return doc_vec, class_vec


def spamTest():
    doc_list, class_list = getFileData()
    vocab_list = createVocabList(doc_list)
    # 文档的数量
    doc_count = len(doc_list)
    # 取文档的前百分之十作为测试数据
    test_count = int(doc_count * 0.1)

    # 训练矩阵和对应的类别
    train_mat = []
    train_class = []
    # 取剩下的文档作文训练矩阵
    for doc_index in range(test_count, doc_count):
        train_mat.append(setOfWords2Vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    # 计算出非垃圾邮件、垃圾邮件中每个词汇的条件概率向量以及垃圾邮件出现的概率
    p0V, p1V, p_spam = trainNB0(train_mat, train_class)

    # 存放测试数据索引的集合
    test_set = []
    # 从测试数据库中随机抽取50用作测试,共5次
    err_rate = 0.0
    for count in range(5):
        for i in range(50):
            test_set.append(int(random.uniform(0, test_count)))
        error_count = 0
        for doc_index in test_set:
            # 将测试文档用词集模型转换成词向量
            word_vec = setOfWords2Vec(vocab_list, doc_list[doc_index])
            # 利用贝叶斯准则进行分类
            result = classifyNB(word_vec, p0V, p1V, p_spam)
            # 统计错误次数
            if result != class_list[doc_index]:
                error_count += 1
            # print(result, doc_index, class_list[doc_index])
        this_rate = float(error_count) / len(test_set)
        err_rate += this_rate
        print(f"第{count+1}次错误率为：{this_rate}")
    print(f"平均错误率是{format(err_rate/5, '.2f')}")


def calcMostFreq(vocab_list, full_text):
    import operator
    freq_dict = {}
    for word in vocab_list:
        freq_dict[word] = full_text.count(word)
    sorted_Freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_Freq[:30]


def localWords(feed1, feed0):
    import feedparser
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries'], feed1['entries']))
    for i in range(min_len):
        word_list = feedparser.textParse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = feedparser.textParse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = createVocabList(doc_list)
    top30_words = calcMostFreq(vocab_list, full_text)
    for word in top30_words:
        if word[0] in vocab_list:
            vocab_list.remove(word[0])
    training_set = range(2*min_len)
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]
    # 训练矩阵和对应的类别
    train_mat = []
    train_class = []

    for doc_index in training_set:
        train_mat.append(setOfWords2Vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0V, p1V, p_spam = trainNB0(train_mat, train_class)
    error_count = 0
    for doc_index in test_set:
        # 将测试文档用词集模型转换成词向量
        word_vec = setOfWords2Vec(vocab_list, doc_list[doc_index])
        # 利用贝叶斯准则进行分类
        result = classifyNB(word_vec, p0V, p1V, p_spam)
        # 统计错误次数
        if result != class_list[doc_index]:
            error_count += 1
        # print(result, doc_index, class_list[doc_index])
    print(f"错误率为：{float(error_count) / len(test_set)}")
    return vocab_list, p0V, p1V
