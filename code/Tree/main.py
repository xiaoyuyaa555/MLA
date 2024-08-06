import trees
import treePlot

lenses_data = trees.file2lenses('./lenses.txt')
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses_tree = trees.createTree(lenses_data, lenses_labels)
trees.storeTree(lenses_tree, 'lensesTree.txt')
lenses_tree_local = trees.grabTree('./lensesTree.txt')
# 使用本地保存的树进行画图和预测
treePlot.createPlot(lenses_tree_local)
result = trees.classify(lenses_tree_local, lenses_labels, ['pre', 'hyper', 'yes', 'normal'])
print(result)
