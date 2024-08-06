import numpy as np
from numpy import linalg as la


def load_ex_data():
    return np.array([
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]
    ])  # Using np.array for convenience


def load_ex_data2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 计算欧式距离
def eclud_sim(a, b):
    return 1.0 / (1.0 + la.norm(a - b))


# 皮尔逊相关系数
def pears_sim(a, b):
    if len(a) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(a, b, rowvar=0)[0][1]


# 余弦相似度
def cos_sim(a, b):
    num = float(a.T * b)
    denom = la.norm(a) * la.norm(b)
    return 0.5 + 0.5 * (num / denom)


# 基于物品相似度的推荐
def stand_est(in_data_mat, user, sim_meas, item):
    # 数据中行为用于，列为物品，n即为物品数目
    n = np.shape(in_data_mat)[1]  # 获取物品数目
    sim_total = 0.0  # 相似度总和
    rat_sim_total = 0.0  # 评分与相似度乘积的总和
    # 用户的第j个物品
    for j in range(n):
        user_rating = in_data_mat[user, j]  # 用户对物品的评分
        if user_rating == 0:
            continue
        # 寻找两个用户都评级的物品
        overLap = np.nonzero(np.logical_and(in_data_mat[:, item].A > 0, in_data_mat[:, j].A > 0))[0]

        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(in_data_mat[overLap, item], in_data_mat[overLap, j])  # 计算相似度

        sim_total += similarity  # 更新相似度总和
        rat_sim_total += sim_total * user_rating  # 更新评分与相似度乘积的总和

    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total  # 返回评分估计值


def svd_est(in_data_mat, user, sim_meas, item):
    n = np.shape(in_data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    U, sigma, VT = la.svd(in_data_mat)
    sig3 = np.mat(np.eye(3) * sigma[:3])  # 化为对角阵
    xformed_items = in_data_mat.T * U[:, :3] * sig3.I  # 构造转换后的物品
    for j in range(n):
        userRating = in_data_mat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = sim_meas(xformed_items[item, :].T, xformed_items[j, :].T)
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * userRating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def recommend(in_data_mat, user, N=3, sim_meas=cos_sim, est_method=stand_est):
    # 找出用户未评分的物品
    unrated_items = np.nonzero(in_data_mat[user, :].A == 0)[1]

    if len(unrated_items) == 0:
        return '这里没有你喜欢吃的'
    item_scores = []
    for item in unrated_items:
        estimated_score = est_method(in_data_mat, user, sim_meas, item)  # 使用指定方法估计用户对物品的评分
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]  # 返回前N个评分最高的物品


def print_mat(in_mat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(in_mat[i, k]) > thresh:
                print(1,end='')
            else:
                print(0,end='')
        print('')


def img_compress(num_sv=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        new_row = []
        for i in range(32):
            new_row.append(int(line[i]))
        myl.append(new_row)
    my_mat = np.mat(myl)
    print("****原始矩阵******")
    print_mat(my_mat, thresh)
    U, sigma, VT = la.svd(my_mat)
    sig_recon = np.mat(np.zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k, k] = sigma[k]
    recon_mat = U[:, :num_sv] * sig_recon * VT[:num_sv, :]
    print(f"****使用{num_sv}个元素重构的矩阵******")
    print_mat(recon_mat, thresh)


if __name__ == '__main__':
    # test1
    # data_arr = load_ex_data()
    # m, n = np.shape(data_arr)
    # U, Sigma, VT = la.svd(data_arr)  # 调用linalg中封装好的svd函数
    # print("Sigma:\n:", Sigma)
    #
    # Sig3 = np.mat(np.array([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]]))
    # print("重构:\n", U[:, :3] * Sig3 * VT[:3, :])

    # test2
    # data_mat = np.mat(load_ex_data())
    # print("欧氏距离")
    # print(eclud_sim(data_mat[:, 0], data_mat[:, 4]))
    # print(eclud_sim(data_mat[:, 0], data_mat[:, 0]))
    #
    # print("皮尔逊相关系数")
    # print(pears_sim(data_mat[:, 0], data_mat[:, 4]))
    # print(pears_sim(data_mat[:, 0], data_mat[:, 0]))
    #
    # print("余弦相似度")
    # print(cos_sim(data_mat[:, 0], data_mat[:, 4]))
    # print(cos_sim(data_mat[:, 0], data_mat[:, 0]))

    # 推荐系统
    # 重新设计矩阵
    # data_mat = np.mat(load_ex_data())
    # data_mat[0, 1] = data_mat[0, 0] = data_mat[1, 0] = data_mat[2, 0] = 4
    # data_mat[3, 3] = 2
    # print(data_mat)
    # print("才用欧氏距离推荐的结果:\n", recommend(data_mat, user=2, sim_meas=eclud_sim))
    # print("才用皮尔逊相关系数推荐的结果:\n", recommend(data_mat, user=2, sim_meas=pears_sim))
    # print("才用余弦相似度推荐的结果:\n", recommend(data_mat, user=2, sim_meas=cos_sim))

    # 利用svd进行评分估计
    # data_mat = np.mat(load_ex_data2())
    # print(data_mat)
    #
    # U, sigma, VT = la.svd(data_mat)
    # sig2 = sigma**2
    # print("总能量的百分之90：", sum(sig2)*0.9)
    # print("前两个元素包含的能量：", sum(sig2[:2]))
    # print("前三个元素包含的能量：", sum(sig2[:3]))
    #
    # print("采用欧氏距离推荐的结果:\n", recommend(data_mat, user=2, sim_meas=eclud_sim, est_method=svd_est))
    # print("采用皮尔逊相关系数推荐的结果:\n", recommend(data_mat, user=2, sim_meas=pears_sim, est_method=svd_est))
    # print("采用余弦相似度推荐的结果:\n", recommend(data_mat, user=2, sim_meas=cos_sim, est_method=svd_est))

    # 基于SVD的图像压缩
    img_compress(2)