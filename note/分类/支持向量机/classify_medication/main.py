import Classifier
import numpy as np

data_list, label_list = Classifier.load_file('./data/drug.txt')
data_arr = np.array(data_list)
num_data = data_arr.shape[0]

test_size = 0.13  # 取数据集的百分之13用作测试
num_test_iter = int(test_size * num_data)
data_list = data_list[num_test_iter: num_data]
label_list = label_list[num_test_iter:num_data]

"""
KNN
"""
# 创建KNN分类器对象，一部分数据集用作训练，一部分用作测试
knn = Classifier.KNN(data_list, label_list, 3)
params = {
    'k': range(3, 5)
}
print(knn.grid_search(params, 42))
# 测试分类器
knn_err_rate = knn.test_function('./data/drug.txt', test_size)
print(f"knn共测试了{num_test_iter}次，总的错误率是：{round(knn_err_rate * 100, 3)}%\n")

"""
SVM
"""
data_list = Classifier.standard_data(data_list)
# 创建SVM分类器对象
svm = Classifier.SVM(data_list, label_list, c=100, toler=1e-4, max_iter=90, k_tup=('rbf', 1))
params = {
    'model': 'ovr',  # 多分类器采用ovr还是ovo策略
    'kernel': ['rbf', 'line'],  # 核函数类型
    'c': range(1, 10, 100),  # 目标函数的惩罚系数
    'gamma': np.arange(1e-2, 1e-1, 1)  # 核函数的系数
}
print(svm.grid_search(params, seed=42))
svm.ovr_trainer1()
svm_err_rate2 = svm.test_function('./data/drug.txt', test_size, 'ovr')
print(f"svm共测试了{num_test_iter}次，总的错误率是：{round(svm_err_rate2 * 100, 3)}%\n")

# svm3 = Classifier.SVM(data_list,label_list, c=100, toler=1e-3, max_iter=100,k_tup=('rbf', 1.5))
# params = {
#     'model':'ovo',
#     'kernel':['rbf', 'line'],
#     'c': [1, 10, 100],
#     'gamma': [0.5, 1, 1.5]
# }
# print(svm1.grid_search(params))
# svm3.ovo_trainer0()
# svm_err_rate3 = svm3.test_function('./data/drug.txt', test_size, 'ovo')
# print(f"svm_ovo 共测试了{num_test_iter}次，总的错误率是：{round(svm_err_rate3, 5) * 100}%\n")

"""
Tree
"""
test_size = 0.14  # 取数据集的百分之ho_ratio用作测试
num_test_iter = int(test_size * num_data)
# 创建决策树分类器对象
data_list, feature_list = Classifier.load_file('./data/drug.txt', 'Tree')
tree = Classifier.Tree(data_list[num_test_iter: num_data], feature_list)

tree_train_err_rate = tree.trainer0()
tree_test_err_rate = tree.test_function('./data/drug.txt', test_size)
print(f"tree训练错误率：{round(tree_train_err_rate * 100, 3)}%\n")
print(f"tree共测试了{num_test_iter}次，总的错误率是：{round(tree_test_err_rate, 5) * 100}%\n")
