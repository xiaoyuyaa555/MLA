# 基于概率论的分类方法

## 朴素贝叶斯

是一种基于贝叶斯决策理论的分类方法，朴素二字表现在做了特征条件独立假设。

|          | 说明                                     |
| -------- | ---------------------------------------- |
| 使用类型 | 标称型                                   |
| 优点     | 在数据较小时仍然有效，可以处理多类别问题 |
| 缺点     | 对输入数据的准备方式极其敏感             |

**贝叶斯决策理论核心思想**

选择具有最高概率的决策，即对于一个新的数据点(x, y)，可以使用以下的规则取判断类别：

- 如果p1(x, y) > p2(x, y)，那么类别为1

- 如果p2(x, y) > p1(x, y)，那么类别为2

p1()  p2()只是简单的描述，在实际中需要利用条件概率进行计算，比较p(c1|x, y)、p(c2|x, y)的大小，即某个(x, y)来自c1,c2的概率，根据贝叶斯准则可以得到：
$$
p(c_i|x, y)=\frac{p(x, y|c_i)p(c_i)}{p(x,y)}
$$
因此，可以得到贝叶斯分类准则为：

- 如果p(c1|x, y) > p(c2|x, y)，那么类别为1

- 如果p(c2|x, y) > p(c1|x, y)，那么类别为2

### 准备数据

#### **从文本构建词向量**

- 首先创建一个包含所有不重复词的列表

- 然后判断输入文档中的词是否出现在列表中，并转换成对应的词向量，在转换时有两种模型：

  - 更关心词汇是否出现的词集模型

  - 更关心词汇出现频率的词袋模型

##### 所有不重复词的列表

```python
# 创建实验样本，斑点狗网站的留言
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
```

##### **词集模型**

也称贝努利模型，不考虑出现次数，只考虑是否出现，即认为假设词是等权重的。适用于以下情况：

- 当文本分类任务中，对于一个词是否出现的信息很重要，而词的出现次数并不重要时。例如，垃圾邮件分类问题中，只需要知道某个特定词是否出现在邮件中，而不关心该词在邮件中出现的次数。

```python
# 检查输入文档种的单词是否在词汇表内
def setOfWords2Vec(vocab_list, inputSet):
    return_vec = [0] * len(vocab_list)
    for word in inputSet:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print(f"the word:{word} is not in my vocabulary!")
    return return_vec
```

##### **词袋模型**

词袋模型也称多项式模型，考虑出现的次数，关心出现的频率。适用于以下情况：

- 当文本分类任务中，对于词语的出现频率和次数信息很重要时。例如，情感分析任务中，一个文档中某些情感词的出现次数可以反映该文档的情感倾向。
- 当文本分类问题中，需要考虑词语之间的相对重要性时。例如，在主题分类任务中，某些特定关键词的出现次数可能比其他词更重要，文档词袋模型可以更好地捕捉到这种信息。

```python
# 检查输入文档种的单词是否在词汇表内
# 词袋模型也称多项式模型，考虑出现的次数，关心出现的频率
def bagOfWords2Vec(vocab_list, inputSet):
    return_vec = [0] * len(vocab_list)
    for word in inputSet:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
        else:
            print(f"the word:{word} is not in my vocabulary!")
    return return_vec
```

执行效果

```python
import bayes

list_of_pos, list_of_classes = bayes.loadDataSet()
my_vocab_list = bayes.createVocabList(list_of_pos)
print(my_vocab_list)

re_vec1 = bayes.setOfWords2Vec(my_vocab_list, list_of_pos[0])
re_vec2 = bayes.setOfWords2Vec(my_vocab_list, list_of_pos[2])
print(re_vec1)
print(re_vec2)

# 输出
# ['dog', 'park', 'my', 'problems', 'garbage', 'licks', 'mr', 'help', 'take', 'food', 'him', 'ate', 'love', 'posting', 'so', 'not', 'stupid', 'flea', 'I', 'dalmation', 'stop', 'steak', 'cute', 'quit', 'to', 'is', 'please', 'maybe', 'worthless', 'buying', 'has', 'how']
# [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### 训练算法

#### **从词向量计算概率**

我们需要通过贝叶斯准则比较词向量属于侮辱类和非侮辱类的概率，用w代表词向量代替(x，y)。
$$
p(c_i|w)=\frac{p(w|c_i)p(c_i)}{p(w)}
$$

- 对于p(c<sub>i</sub>)可以用非侮辱或侮辱类文档数除以总的文档数得到

- 将词向量 w 展开成一个个独立的特征，则
  $$
  p(w|c_i)=p(w_0,w_1..w_n|c_i)=p(w_0|c_i)p(w_1|c_i)...p(w_n|c_i)
  $$
  在计算时先确认标签是属于哪一个类别，再统计各特征出现的次数，最后用各特征出现的总次数除以总词数

- 对于p(w)，即 w 发生的概率。贝叶斯准则的分子表示某个类别 i 中 w 发生的概率，而p(w)表示所有类别中w发生的概率之和，因此p(w)是一个归一化因子，其目的是保证后验概率之和为1。在计算先验概率是并不需要知道它的值，故不计算。

```python
def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    # 认为每一篇文档的各个特征都是一样的
    num_words = len(train_matrix[0])
    # 计算含侮辱性词汇的概率
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = zeros(num_words)
    p1_num = zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:                 # 含侮辱性词汇
            p1_num += train_matrix[i]              # 将侮辱性词汇个数+1
            p1_denom += sum(train_matrix[i])       # class1总词汇数增加
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p0_vec = p0_num / p0_denom                     # 对应词汇个数/总词汇数
    p1_vec = p1_num / p1_denom
    return p0_vec, p1_vec, p_abusive
```

### 测试算法

**根据实际情况修改分类器**
$$
p(w|c_i)=p(w_0,w_1..w_n|c_i)=p(w_0|c_i)p(w_1|c_i)...p(w_n|c_i)
$$
由这个公式可以知道，我们需要通过乘积的形式计算某一类别中 w 发生的概率，而在`trainNB0()`函数输出的向量中，会存在很多0元素，这会导致最后的计算结果为0，因此需要对此进行调整，初始化每个词出现的次数为1

```python
p0_num = ones(num_words)
p1_num = ones(num_words)
```
由于因子都很小，因此采用取对数的方式避免下溢

```python
p0_vec = log(p0_num / p0_denom)                     
p1_vec = log(p1_num / p1_denom)
```

**测试函数**

```python
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
    test_post1 = ['stupid', 'garbage']
    test_doc = array(setOfWords2Vec(my_vocab_list, test_post1))
    print(f"{test_post1} classified as: {classifyNB(test_doc, p0_vec, p1_vec, p_abusive)}")
```

测试结果

```python
import bayes

# 0表示无侮辱词
# 1表示有侮辱词
bayes.testingNB()

# 结果
# ['love', 'my', 'dalmation'] classified as: 0
# ['stupid', 'garbage'] classified as: 1
```

### 示例：过滤垃圾邮件

准备数据

```
将文本转换成列表
对文档的每一行：
	通过制表符划分成标签和邮件内容两部分
	对标签部分：
		如果标签为'ham'，记为0 -> 追加进类别列表
		如果标签为'spam'，记为1 -> 追加进类别列表
	对邮件内容部分：
		用正则表达式划分邮件内容
		将大写字母改为小写字母
		将修改后的内容存放进文档列表
返回文档列表和类别列表

createVocabList函数将返回文档列表转换为词向量
```

**测试完整代码**

```python
import re
def getFileData():
    f = open('./email.txt', 'r', encoding='UTF-8')
    class_vec = []
    doc_vec = []
    for email_lines in f:
        # 先按照制表符划分成标签和邮件内容两部分
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
```

**运行结果**

```python
import bayes

bayes.spamTest()

# 结果
# 第1次错误率为：0.06
# 第2次错误率为：0.05
# 第3次错误率为：0.04
# 第4次错误率为：0.035
# 第5次错误率为：0.032
# 平均错误率是0.04
```

### 示例：从个人广告中获得区域倾向

