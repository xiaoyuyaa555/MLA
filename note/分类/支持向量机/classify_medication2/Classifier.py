# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def file2data(file_path, test_size, random_state=42):
    """将数据集转换成训练集、测试集

    Parameters
    ----------
    file_path :
        文件路径.

    test_size : float
        测试集占比

    random_state : int or None
        随机数种子

    Returns
    -------
    x_train, x_test, y_train, y_test :
        返回特征和标签

        x_train, x_test, y_train, y_test
    """

    if test_size > 1 or test_size < 0:
        raise '输入的测试比例错误'

    # 加载数据集
    df = pd.read_csv(file_path, engine="python")

    # LabelEncoder对特征进行数字编码
    label_encoder = LabelEncoder()
    for cow in df:
        if type([cow][0]).__name__ == 'str':
            df[cow] = label_encoder.fit_transform(df[cow])

    # 将数据分为特征信息（X）和标签（Y）
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    # train_test_split 函数进行数据集划分
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # StandardScaler 将特征缩放到均值为0，标准差为1的范围内
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    return x_train, x_test, y_train, y_test


def accuracy_function(model, x_test, y_test, report=False):
    """用测试集测试模型

    Parameters
    ----------
    model :
        用于测试的模型

    x_test :
        测试集的特征数据

    y_test :
        测试集的标签

    report :
        是否打印分类报告，默认为 False

    Returns
    -------
    test_accuracy :
        模型在测试集上的准确率
    """
    from sklearn.metrics import classification_report, accuracy_score
    # 在测试集上进行预测
    test_predictions = model.predict(x_test)

    # 在测试集上评估模型
    test_accuracy = accuracy_score(y_test, test_predictions)

    # 打印分类报告
    if report:
        print("\n分类报告:\n", classification_report(y_test, test_predictions))

    return test_accuracy


if __name__ == '__main__':
    data_train, data_test, label_train, label_test = file2data('./drug200.csv', 0.3)

    # 创建SVM模型
    svm = SVC(kernel='linear', C=1.0, gamma='auto')
    # 定义参数网格
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': range(1, 10),
        'gamma': [1e-3, 1e-2, 1e-1, 1]
    }
    # 使用网格搜索
    svm = GridSearchCV(svm, param_grid, cv=5)
    svm.fit(data_train, label_train)

    print("SVM 测试准确性:", round(accuracy_function(svm, data_test, label_test), 5) * 100, '%')

    # KNN模型
    knn = KNeighborsClassifier()
    knn.fit(data_train, label_train)
    # 定义参数网格
    param_grid = {
        'n_neighbors': range(1, 10)
    }
    # 使用网格搜索
    knn = GridSearchCV(knn, param_grid, cv=5)
    knn.fit(data_train, label_train)

    print("KNN 测试准确性:", round(accuracy_function(knn, data_test, label_test), 5) * 100, '%')
