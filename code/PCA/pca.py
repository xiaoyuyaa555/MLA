from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def load_data_set(file_name, delim="\t"):
    with open(file_name) as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
        dat_arr = [list(map(float, line)) for line in string_arr]
    return mat(dat_arr)


# PAC算法
def pca(data_mat, top_n_feat=9999999):
    mean_vals = mean(data_mat, axis=0)              # 计算数据集的均值
    mean_removed = data_mat - mean_vals             # 数据集去中心化
    cov_mat = cov(mean_removed, rowvar=0)           # 计算协方差矩阵
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))     # 计算协方差矩阵的特征值和特征向量
    eig_val_ind = argsort(eig_vals)                 # 对特征值进行排序，
    eig_val_ind = eig_val_ind[:-(top_n_feat + 1):-1]    # 得到前top_n_feat个特征值的索引
    red_eig_vects = eig_vects[:, eig_val_ind]       # 取对应的特征向量
    low_d_data_mat = mean_removed * red_eig_vects   # 转换到新空间
    recon_mat = (low_d_data_mat * red_eig_vects.T) + mean_vals      # 重构数据集
    return low_d_data_mat, recon_mat


# 用平均数值代替NaN值
def replace_nan_with_mean(file_name):
    dat_mat = load_data_set(file_name, ' ')
    num_feat = shape(dat_mat)[1]
    for i in range(num_feat):
        mean_val = mean(dat_mat[nonzero(~isnan(dat_mat[:, i].A))[0], i])  # 非 NaN 值的均值
        dat_mat[nonzero(isnan(dat_mat[:, i].A))[0], i] = mean_val  # 将 NaN 值替换为均值
    return dat_mat


if __name__ == '__main__':
    # test1
    # data_mat = load_data_set("testSet.txt")
    # low_d_mat, recon_mat = pca(data_mat, 2)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker="^", s=90)
    # ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker="o", s=50, c="red")
    #
    # d_mat, recon_mat = pca(data_mat, 1)
    # ax = fig.add_subplot(212)
    # ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker="^", s=90)
    # ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker="o", s=50, c="red")
    #
    # plt.show()

    # test2
    data_mat = replace_nan_with_mean('secom.data')
    mean_vals = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_mat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))
    # print eigVals
    print('方差总和：', sum(eig_vals))
    print(f'前1的方差和：{sum(eig_vals[:1])},占比为:{round((sum(eig_vals[:1]) / sum(eig_vals)) * 100, 3)}%')
    print(f'前3的方差和：{sum(eig_vals[:3])},占比为:{round((sum(eig_vals[:3]) / sum(eig_vals)) * 100, 3)}%')
    print(f'前6的方差和：{sum(eig_vals[:6])},占比为:{round((sum(eig_vals[:6]) / sum(eig_vals))*100,3)}%')
    print(f'前20的方差和：{sum(eig_vals[:20])},占比为:{round((sum(eig_vals[:20]) / sum(eig_vals)) * 100, 3)}%')
    plt.plot(eig_vals[:20])  # 对前20个画图观察
    plt.show()




