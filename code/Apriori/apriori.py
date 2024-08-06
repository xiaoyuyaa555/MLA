from numpy import *


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 生成候选项集 C1
def create_c1(data_set):
    c1 = []  # 用于存储所有单个商品
    for transaction in data_set:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])

    c1.sort()
    return list(map(frozenset, c1))  # frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用


# 发现频繁项集
def scan_d(d, ck, min_support):
    ss_cnt = {}  # 选项集的支持度计

    # 遍历数据集中的每个交易和候选项集
    for tid in d:
        for can in ck:
            # 如果候选集包含在交易中，对它进行计数
            if can.issubset(tid):
                if can not in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1

    num_items = float(len(d))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:  # 选择大于最小支持度的项集
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


# 输出所有可能的候选项集 Ck
# 例如lk={0}{1}{2},若k=2,则Ck={0,1}{0,2}{1,2};若k=3,则Ck={0,1,2}
def apriori_gen(lk, k):
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:  # if first k-2 elements are equal
                ret_list.append(lk[i] | lk[j])  # set union
    return ret_list


# Apriori 算法
def apriori(data_set, min_support=0.5):
    c1 = create_c1(data_set)  # 生成初始候选项集 C1
    d = list(map(set, data_set))  # 将数据集中的每个事务映射为集合
    l1, support_data = scan_d(d, c1, min_support)  # 扫描数据集，得到频繁项集 l1和其对应的支持度
    l = [l1]
    k = 2

    # 循环生成频繁 k-项集
    while len(l[k - 2]) > 0:  # 上一次构建出的集合CN的LN长度大于0
        ck = apriori_gen(l[k - 2], k)
        lk, sup_k = scan_d(d, ck, min_support)
        support_data.update(sup_k)  # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        l.append(lk)
        k += 1
    return l, support_data


# 生成关联规则
def generate_rules(l, support_data, min_conf=0.7):
    big_rule_list = []
    for i in range(1, len(l)):
        # 获取频繁项集中每个组合的所有元素
        for freq_set in l[i]:
            H1 = [frozenset([item]) for item in freq_set]  # 转换为只包含单个元素的集合列表
            if i > 1:  # 两个以上的组合
                rules_from_conseq(freq_set, H1, support_data, big_rule_list, min_conf)
            else:  # 只有两个的组合
                calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list


# 计算可信度
def calc_conf(freq_set, H, support_data, brl, min_conf=0.7):
    pruned_H = []
    for conseq in H:
        # 关联规则的置信度
        # freq_set - conseq 表示的是从频繁项集 freq_set 中包含在规则右部 conseq 中的元素去除，得到的新集合
        # 例如，freq_set={2, 3},conseq={3}，则规则为2->3,因此计算的是support(2,3)/support(2)
        # freq_set - conseq 表示的正是从{2,3}中去除3
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freq_set - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H


# 递归计算频繁项集的规则
def rules_from_conseq(freq_set, H, support_data, brl, min_conf=0.7):
    m = len(H[0])
    if len(freq_set) > (m + 1):  # 保证右部元素组合的长度能够继续增加
        hmp1 = apriori_gen(H, m + 1)  # 生成新的候选项集
        hmp1 = calc_conf(freq_set, hmp1, support_data, brl, min_conf)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)


"""
国会投票代码
"""
# from time import sleep
# import votesmart
# def get_action_ids():
#     action_id_list = []
#     bill_title_list = []
#     votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#     with open('recent20bills.txt') as fr:
#         for line in fr.readlines():
#             bill_num = int(line.split('\t')[0])
#             try:
#                 bill_detail = votesmart.votes.getBill(bill_num)  # API call
#                 for action in bill_detail.actions:
#                     if action.level == 'House' and \
#                     (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                         action_id = int(action.actionId)
#                         print('bill: %d has actionId: %d' % (bill_num, action_id))
#                         action_id_list.append(action_id)
#                         bill_title_list.append(line.strip().split('\t')[1])
#             except Exception as e:
#                 print("problem getting bill %d: %s" % (bill_num, e))
#             sleep(1)  # Delay to be polite
#     return action_id_list, bill_title_list
#
# def get_trans_list(action_id_list, bill_title_list):
#     item_meaning = ['Republican', 'Democratic']  # List of what each item stands for
#     for bill_title in bill_title_list:  # Fill up item_meaning list
#         item_meaning.append('%s -- Nay' % bill_title)
#         item_meaning.append('%s -- Yea' % bill_title)
#     trans_dict = {}  # List of items in each transaction (politician)
#     vote_count = 2
#     for action_id in action_id_list:
#         sleep(3)
#         print('getting votes for actionId: %d' % action_id)
#         try:
#             vote_list = votesmart.votes.getBillActionVotes(action_id)
#             for vote in vote_list:
#                 if vote.candidateName not in trans_dict:
#                     trans_dict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         trans_dict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         trans_dict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     trans_dict[vote.candidateName].append(vote_count)
#                 elif vote.action == 'Yea':
#                     trans_dict[vote.candidateName].append(vote_count + 1)
#         except Exception as e:
#             print("problem getting actionId: %d: %s" % (action_id, e))
#         vote_count += 2
#     return trans_dict, item_meaning


if __name__ == '__main__':
    # test1
    # data_set = load_data_set()
    # l1, support_data0 = scan_d(data_set, create_c1(data_set), 0.5)
    # print(l1)
    # test2
    # l, support_data1 = apriori(data_set, 0.5)
    # rules = generate_rules(l, support_data1,0.5)

    # 毒蘑菇
    mush_data_set = [line.split() for line in open('mushroom.dat', encoding='utf-8').readlines()]
    mush_l, mush_support = apriori(mush_data_set, min_support=0.3)
