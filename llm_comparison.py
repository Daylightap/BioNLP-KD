import matplotlib
# 【关键修改】必须在 import pyplot 之前设置
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ===========================
# 数据准备
# ===========================
data = [
    {"name": "DeepSeek-V3.2",           "params": 671,  "f1": 0.4529},
    {"name": "DeepSeek-V3.1-Terminus",  "params": 685,  "f1": 0.4364},
    {"name": "Qwen3-Next-80B-A3B",      "params": 80,   "f1": 0.4053},
    {"name": "Kimi-K2-Instruct",        "params": 1000, "f1": 0.4015},
    {"name": "Qwen2.5-72B",             "params": 72,   "f1": 0.3966}
]

# 提取数据
names = [d["name"] for d in data]
params = [d["params"] for d in data]
f1_scores = [d["f1"] for d in data]

# ===========================
# 绘图设置
# ===========================
plt.figure(figsize=(10, 6), dpi=300)

# 定义颜色 (使用 tab10 调色板)
colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

# 定义气泡大小 (将参数量放大以便视觉观察)
sizes = [p * 3 for p in params]

# 绘制散点
for i in range(len(names)):
    plt.scatter(
        params[i],
        f1_scores[i],
        s=sizes[i],        # 大小基于参数量
        color=colors[i],   # 颜色
        alpha=0.7,         # 透明度
        edgecolors='black', # 边缘颜色
        linewidth=1,       # 边缘宽度
        label=f"{names[i]} ({params[i]}B)" # 图例
    )

    # 添加文字标注
    y_offset = 0.003
    plt.text(params[i], f1_scores[i] + y_offset, names[i],
             fontsize=9, ha='center', va='bottom', fontweight='bold', alpha=0.8)

# ===========================
# 装饰图表
# ===========================
plt.title("F1 Score vs Model Parameters (Zero-shot NER)", fontsize=14, pad=20)
plt.xlabel("Model Parameters (Billions)", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)

# 设置坐标轴范围
plt.xlim(0, 1100)
plt.ylim(0.38, 0.47)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# ---------------------------
# 图例设置
# ---------------------------
# 1. 位置改为 'upper right' (右上角)
lgnd = plt.legend(title="Models", title_fontsize=10, fontsize=9, loc='upper right', frameon=True)

# 2. 【关键修正】使用 legend_handles 替代 legendHandles
# 统一图例中圆点的大小为固定值 100
for handle in lgnd.legend_handles:
    handle.set_sizes([100.0])

# 紧凑布局
plt.tight_layout()

# ===========================
# 保存
# ===========================
save_path = "f1_vs_params_bubble_chart.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"图表已保存至: {save_path}")