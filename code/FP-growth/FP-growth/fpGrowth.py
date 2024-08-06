class tree_node:
    def __init__(self, name, num_occur, parent_node):
        self.name = name  # 节点名称
        self.count = num_occur  # 计数
        self.node_link = None  # 链接到相似节点的链接
        self.parent = parent_node  # 父节点
        self.children = {}  # 子节点的字典

    def inc(self, num_occur):
        self.count += num_occur  # 计数增加

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def create_tree(data_set, min_sup=1):
    header_table = {}  # 头表，存储频繁项及其出现次数
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]

    # 删除不满足最小支持度的项
    for k in list(header_table.keys()):
        if header_table[k] < min_sup:
            header_table.pop(k)

    # 剩余频繁项集
    freq_item_set = set(header_table.keys())
    # 如果频繁项集为空，则返回空
    if len(freq_item_set) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]  # 初始化头表项
    ret_tree = tree_node('Null Set', 1, None)  # 创建根节点
    for tran_set, count in data_set.items():
        local_dataset = {}
        for item in tran_set:
            if item in freq_item_set:
                local_dataset[item] = header_table[item][0]  # 获取本地数据集
        if len(local_dataset) > 0:
            ordered_items = [v[0] for v in sorted(local_dataset.items(), key=lambda p: p[1], reverse=True)]  # 排序本地数据集
            update_tree(ordered_items, ret_tree, header_table, count)  # 更新树
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = tree_node(items[0], count, in_tree)  # 创建新节点
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = in_tree.children[items[0]]  # 更新头表
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])

    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node  # 更新链表


def load_simp_dat():
    simp_data = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                 ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simp_data


def create_init_set(data_set):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


def ascend_tree(leaf_node, pre_fix_path):
    # 如果叶子节点有父节点，则向上遍历至根节点，并记录路径
    if leaf_node.parent is not None:
        pre_fix_path.append(leaf_node.name)  # 将当前节点名称添加到路径中
        ascend_tree(leaf_node.parent, pre_fix_path)  # 递归向上遍历父节点


# 生成条件模式基
def find_pre_fix_path(base_pat, tree_node):
    cond_pats = {}  # 条件模式基
    while tree_node is not None:
        pre_fix_path = []  # 前缀路径
        ascend_tree(tree_node, pre_fix_path)  # 调用ascend_tree函数获取路径
        if len(pre_fix_path) > 1:
            cond_pats[frozenset(pre_fix_path[1:])] = tree_node.count  # 存储以给定元素项为结尾的路径和对应的计数
        tree_node = tree_node.node_link  # 移动到下一个相似节点
    return cond_pats


# 递归查找频繁项集的minTree
def mine_tree(in_tree, header_table, min_sup, pre_fix, freq_item_list):
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]  # 根据头表项的计数排序

    # 从低到高排序之后，遍历bigL中的每一项，即自底向上
    for base_pat in bigL:
        new_freq_set = pre_fix.copy()  # 复制前缀
        new_freq_set.add(base_pat)  # 添加当前项作为新的频繁项集
        freq_item_list.append(new_freq_set)  # 将新的频繁项集添加到列表中

        # 根据当前项生成条件模式基
        cond_patt_bases = find_pre_fix_path(base_pat, header_table[base_pat][1])
        # 创建条件FP树
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_sup)
        if my_head is not None:
            # 递归挖掘条件FP树
            mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)


"""
twitter
"""
# import twitter
# from time import sleep
# import re
#
#
# def textParse(bigString):
#     urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
#     listOfTokens = re.split(r'\W*', urlsRemoved)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
#
# def getLotsOfTweets(searchStr):
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     # you can get 1500 results 15 pages * 100 per page
#     resultsPages = []
#     for i in range(1, 15):
#         print("fetching page %d" % i)
#         searchResults = api.GetSearch(searchStr)
#         resultsPages.append(searchResults)
#         sleep(6)
#     return resultsPages
#
#
# def mineTweets(tweetArr, minSup=5):
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text))
#     initSet = create_init_set(parsedList)
#     myFPtree, myHeaderTab = create_tree(initSet, minSup)
#     myFreqList = []
#     mine_tree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#     return myFreqList


if __name__ == '__main__':
    # 简单数据集
    # simp_dat = load_simp_dat()
    # init_set = create_init_set(simp_dat)
    # my_fp_tree, my_heat_tabel = create_tree(init_set, 3)
    # my_fp_tree.disp()

    # 挖掘频繁项目
    # simp_dat = load_simp_dat()
    # init_set = create_init_set(simp_dat)
    # my_fp_tree, my_heat_tabel = create_tree(init_set, 3)
    # freq_items = []
    # mine_tree(my_fp_tree, my_heat_tabel, 3, set([]), freq_items)
    # print("Frequent items:")
    # print(freq_items)

    # twitter
    # lots_tweets = getLotsOfTweets('RIMM')
    # print(lots_tweets[0][4].text)
    # list_of_terms = mineTweets(lots_tweets, 20)
    # print(len(list_of_terms))
    # print("Terms list:")
    # print(list_of_terms)

    # 从新闻网站点击流中发掘
    with open('kosarak.dat') as f:
        parse_dat = [line.split() for line in f.readlines()]
    init_set = create_init_set(parse_dat)
    my_fp_tree, my_heat_tabel = create_tree(init_set, 100000)  # 查看超过十万人浏览过的新闻
    freq_list = []
    mine_tree(my_fp_tree, my_heat_tabel, 100000, set([]), freq_list)
    print(len(freq_list))
    print("Frequent list:")
    print(freq_list)
