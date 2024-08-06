# SVM如何解决多分类问题

SVM是一类用于解决二分类问题的机器学习算法，它的目的是找到一个最优的超平面，将两类不同的样本尽可能的分隔开。在多分类问题上，基本思路就是将其转换成多个二分类问题，常见的方法有：**一对多（One-vs-Rest，OVR）**，**一对一（One-vs-One，OVO）**。（）

## 一对多 OVR

一对多的意思就是，以一类样本为正类，其余类样本为负类进行区分，因此对于共有K个类别的多分类问题，需要K个SVM分类器。在分类时，将未知样本分类为具有最大分类函数值的那类。

如：现在有A, B, C ,D四类标签需要分类，在训练时就需要分成4个SVM分类器：

- A为正类，B, C, D为负类
- B为正类，A, C, D为负类
- C为正类，A, B, D为负类
- D为正类，A, B, C为负类

在训练时，对这四个分类器进行训练，然后得到四个训练结果。

在测试时，将测试向量分别输入四个分类器进行测试，可以得到四个测试结果$f_1(x)$、$f_2(x)$、$f_3(x)$、$f_4(x)$，最后选择测试结果最大的一个作为分类的结果。

```python
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

        # 创建一个新的SVM
        svm_multi_classifier = SVM(self.data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)
        svm_multi_classifier.trainer0()
        self.multi_classifier_list['ovr'].append(svm_multi_classifier)
```

由于只有一个类别被标记为正类，其余的类别都被标记为负类，这样会导致负类样本数量远远多于正类样本数量。这种样本不平衡可能导致分类器对负类样本更加敏感，忽视正类样本，从而影响分类的准确性。

为了解决这个问题，首先要平衡正类和负类训练样本数量，然后为了增加负类样本的多样性，从未使用的负类中随机抽取一部分作为训练样本。通过这样的方式在保证数量相对平衡的状态下，保证负类样本的多样性。	

举个例子：

1. 假设A类有100个样本，B类有200个样本，C类有200个样本。我们决定采用50%的B类样本和50%的C类样本作为负类。
2. 那么训练集中的正类样本即为A类样本（100个），负类样本为B类样本的50%（100个）和C类样本的50%（100个）。
3. 接下来，我们可以从剩余的B类样本和C类样本中分别随机抽取一部分样本，例如从剩余的B类样本中抽取50个样本，从剩余的C类样本中抽取50个样本，将它们作为额外的负类数据加入到训练集中。

这样，我们就构建了一个更加平衡的训练集，其中正类样本为A类样本（100个），负类样本由B类样本的50%（100个）、C类样本的50%（100个）以及额外抽取的B类样本（50个）和C类样本（50个）组成。

```python
def ovr_trainer1(self):
    # 对每个类别
    for iter in self.groups:
        remain_data_index = []
        data_count = {label: 1 for label in self.groups}
        data_mat = []
        label_mat = []

        # 对样本中数量较少的类别
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

        # 创建一个新的SVM
        svm_multi_classifier = SVM(data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)
        svm_multi_classifier.trainer0()
        self.multi_classifier_list['ovr'].append(svm_multi_classifier)
```



## 一对一 OVO

一对一的意思就是，对K个类别的样本进行$C_k^2$两两分类，因此共有${K*(K-1)}/{2}$个分类器。在分类时，将未知样本分类为得票最多的那类。

如：现在有A, B, C ,D四类标签需要分类，在训练时就需要分成6个SVM分类器：

- A为正类，B为负类
- A为正类，C,为负类
- A为正类，D为负类
- B为正类，C为负类
- B为正类，D为负类
- C为正类，D为负类

在训练时，对这6个分类器进行训练，然后得到6个训练结果。

在测试的时候，把对应的向量分别对六个结果进行测试，然后采取投票形式，最后得到一组结果。即在（A，B）分类时，结果为A则A加一票；（A，C）分类时，结果为C则C加一票；最后取票数最多的为分类结果。

这种方法的代价是分类器数量太大，大部分情况下其数量都大于OVR。

```python
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
        svm_multi_classifier = SVM(self.data_mat, label_mat, self.c, self.toler, self.max_iter, self.k_tup)
        svm_multi_classifier.trainer0()
        self.multi_classifier_list['ovo'].append(svm_multi_classifier)
```

