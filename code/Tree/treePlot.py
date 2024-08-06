import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 确定叶子节点数
def getNUmLeafs(myTree: dict):
    num_leaf = 0
    first_str = list(myTree.keys())[0]
    second_dir = myTree[first_str]
    # 对每一个子节点进行查询，如果类型为dict则为非叶子节点，否则则为叶子节点
    for key in second_dir.keys():
        if type(second_dir[key]).__name__ == 'dict':
            num_leaf += getNUmLeafs(second_dir[key])
        else:
            num_leaf += 1
    return num_leaf


# 确定树的层数
def getTreeDepth(myTree):
    max_dep = 0
    # python3中不支持直接myTree.keys()进行索引，需要转换成元组或者列表
    first_str = list(myTree.keys())[0]
    second_dir = myTree[first_str]
    # 对每一个子节点进行查询，如果类型为dict则为非叶子节点，继续向下搜索，否则为叶子节点，深度为1
    for key in second_dir.keys():
        if type(second_dir[key]).__name__ == 'dict':
            this_dep = 1 + getTreeDepth(second_dir[key])
        else:
            this_dep = 1
        if this_dep > max_dep:
            max_dep = this_dep
    return max_dep


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",
                            xytext=centerPt, textcoords="axes fraction",
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    x_mid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    y_mid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.axl.text(x_mid, y_mid, txtString)


def plotTree(myTree, parentPR, nodeTxt):
    num_leaf = getNUmLeafs(myTree)
    depth = getTreeDepth(myTree)
    first_str = list(myTree.keys())[0]
    # 在计算节点的位置时，单位长度是 1 / 总的宽或者深
    cntr_ptr = (plotTree.xOff +
                (1.0 + float(num_leaf)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 在设置初始xOff时是-0.5个单位长度，因此这里有个1.0
    plotMidText(cntr_ptr, parentPR, nodeTxt)
    plotNode(first_str, cntr_ptr, parentPR, decision_node)
    second_dict = myTree[first_str]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in second_dict.keys():
        # 如果是非叶子节点，递归调用plotTree
        if type(second_dict[key]).__name__ == 'dict':
            plotTree(second_dict[key], cntr_ptr, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(second_dict[key], (plotTree.xOff, plotTree.yOff), cntr_ptr, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntr_ptr, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    # fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNUmLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.axl = plt.subplot(111, frameon=False)
#     plotNode('decision_node', (0.5, 0.1), (0.1, 0.5), decision_node)
#     plotNode("leaf_node", (0.8, 0.1), (0.3, 0.8), leaf_node)
#     plt.show()


# 创建了两个样本
def retrieveTree(i):
    tree_list = [
                 {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                 {'no surfacing': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}
                ]
    return tree_list[i]
