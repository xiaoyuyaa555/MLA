from svm_kerbel import *
from numpy import *


# 将数字图像信息转换为矩阵
def img2vector(filename):
    return_vect = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(lineStr[j])
    return return_vect


# 将手写识别问题转换为二分类问题，数字9类别为-1，不是数字9类别为1
def loadImage(dirName):
    from os import listdir
    hw_labels = []
    # 获取所有文件
    training_file_list = listdir(dirName)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        training_mat[i, :] = img2vector(dirName + '/' + file_name_str)
    return training_mat, hw_labels


def atestDigits(k_tup=('rbf', 10)):
    data_arr, label_arr = loadImage('../KNN/digits/trainingDigits')
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 10000, k_tup)
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    sv_ind = nonzero(alphas.A > 0)[0]
    s_vs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print(f"这里有{shape(s_vs)[0]}支持向量")
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"训练错误率{(float(error_count) / m)}")

    data_arr, label_arr = loadImage('../KNN/digits/testDigits')
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        predict = kernel_eval.T * multiply(label_sv, alphas[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print(f"测试错误率{(float(error_count) / m)}")


if __name__ == "__main__":
    atestDigits()

