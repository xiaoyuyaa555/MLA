import numpy as np
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import KNN

mat, label = KNN.file2mat("./hailun.txt")
# 设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=14)
# 将fig画布分隔成1行3列,不共享x轴和y轴,fig画布的大小为(13,8)
# 当nrow=1,nclos=3时,代表fig画布被分为三个区域,axs[0]表示第一行第一个区域
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(13, 8))
count_label = len(label)
label_colors = []

# 给不同类型设置颜色
for i in label:
    # 不喜欢是黑色
    if 1 == i:
        label_colors.append('black')
    # 一般有魅力是橘色
    if 2 == i:
        label_colors.append('orange')
    # 很有魅力是红色
    if 3 == i:
        label_colors.append('red')

# 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
axs[0].scatter(x=mat[:, 0], y=mat[:, 1], color=label_colors, s=15, alpha=.5)
# 设置标题,x轴label,y轴label
axs0_title_text = axs[0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', fontproperties=font)
axs0_xlabel_text = axs[0].set_xlabel('每年获得的飞行常客里程数', fontproperties=font)
axs0_ylabel_text = axs[0].set_ylabel('玩视频游戏所消耗时间占', fontproperties=font)
plt.setp(axs0_title_text, size=9, weight='bold', color='red')
plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

# 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
axs[1].scatter(x=mat[:, 0], y=mat[:, 2], color=label_colors, s=15, alpha=.5)
# 设置标题,x轴label,y轴label
axs1_title_text = axs[1].set_title('每年获得的飞行常客里程数与每周消费的冰激淋公升数', fontproperties=font)
axs1_xlabel_text = axs[1].set_xlabel('每年获得的飞行常客里程数', fontproperties=font)
axs1_ylabel_text = axs[1].set_ylabel('每周消费的冰激淋公升数', fontproperties=font)
plt.setp(axs1_title_text, size=9, weight='bold', color='red')
plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

# 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
axs[2].scatter(x=mat[:, 1], y=mat[:, 2], color=label_colors, s=15, alpha=.5)
# 设置标题,x轴label,y轴label
axs2_title_text = axs[2].set_title('玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', fontproperties=font)
axs2_xlabel_text = axs[2].set_xlabel('玩视频游戏所消耗时间占比', fontproperties=font)
axs2_ylabel_text = axs[2].set_ylabel('每周消费的冰激淋公升数', fontproperties=font)
plt.setp(axs2_title_text, size=9, weight='bold', color='red')
plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

# 设置图例
didntLike = mlines.Line2D([], [], color='black', marker='.',
                          markersize=6, label='didntLike')
smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                           markersize=6, label='smallDoses')
largeDoses = mlines.Line2D([], [], color='red', marker='.',
                           markersize=6, label='largeDoses')
# 添加图例
axs[0].legend(handles=[didntLike, smallDoses, largeDoses])
axs[1].legend(handles=[didntLike, smallDoses, largeDoses])
axs[2].legend(handles=[didntLike, smallDoses, largeDoses])

# 展示散点图
plt.show()
