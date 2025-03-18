import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置全局样式
OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 16  # 横坐标字体大小
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 11  # 减小图例字体大小
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['font.family'] = OPT_FONT_NAME
plt.rcParams['pdf.fonttype'] = 42

# 数据
labels = ['Random', 'GloVe', 'FastText', 'Word2Vec']
val_acc = [28.41, 26.25, 24.23, 21.91]
test_acc = [25.97, 20.49, 25.24, 21.29]

# 颜色和样式
BAR_WIDTH = 0.35  # 柱子宽度

# 创建图表
plt.figure(figsize=(8, 5))
x = np.arange(len(labels))

# 绘制柱状图
plt.bar(x - BAR_WIDTH/2, val_acc, BAR_WIDTH, label='Validation Accuracy', color='b', alpha=0.8)
plt.bar(x + BAR_WIDTH/2, test_acc, BAR_WIDTH, label='Test Accuracy', color='r', alpha=0.8)

# 设置标签和标题
plt.xlabel('Model Configuration', fontproperties=LABEL_FP)
plt.ylabel('Accuracy (%)', fontproperties=LABEL_FP)
plt.xticks(x, labels, fontproperties=TICK_FP, rotation=0)  # 横坐标不倾斜
plt.yticks(fontproperties=TICK_FP)
plt.ylim(0, 30)  # Y轴范围0-30%

# 添加图例，保持右上角位置
plt.legend(prop=LEGEND_FP, loc='upper right')  # 使用默认位置，字体减小后应不覆盖

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局
plt.tight_layout()
plt.savefig('task1_vs_task2_bar_chart.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()