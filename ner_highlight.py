import os
import re
from itertools import cycle
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PIL import Image, ImageDraw, ImageFont


# === 配置区域 ===
MODEL_DIR = "./ner_output/biobert_full_model"   # 微调 BioBERT 模型目录
SENTENCE = "The PPR-SMR Protein ATP4 Is Required for Editing the Chloroplast rps8 mRNA in Rice and Maize."  # 样例句子
OUTPUT_IMAGE = "ner_highlight.png"              # 输出图片文件名（仅 ASCII）
FONT_PATHS_TRY = [
    # 按顺序尝试加载的字体路径（根据你的系统调整）
    "/System/Library/Fonts/PingFang.ttc",                # macOS
    "C:/Windows/Fonts/simhei.ttf",                       # Windows (黑体)
    "C:/Windows/Fonts/msyh.ttc",                         # Windows (微软雅黑)
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
]
FONT_SIZE = 28
MAX_WIDTH = 1200               # 图片中文本区域最大宽度（像素）
LINE_SPACING = 12              # 行距（像素）
PADDING = 30                   # 边距（像素）


def load_font() -> ImageFont.FreeTypeFont:
    for fp in FONT_PATHS_TRY:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, FONT_SIZE)
            except Exception:
                pass
    # 回退：Pillow 自带字体
    return ImageFont.load_default()


def run_ner(text: str) -> List[Dict]:
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # 合并连续 token
        device=device
    )
    results = ner(text)
    # 规范化实体标签（去掉可能的 B-/I- 前缀）
    for r in results:
        label = r.get("entity_group", r.get("entity", ""))
        label = re.sub(r"^[B|I|O]-", "", label)
        r["entity_group"] = label
    return results


def build_char_spans(text: str, ents: List[Dict]) -> List[Tuple[int, int, str]]:
    """把 pipeline 返回的实体列表转成 (start, end, label) 三元组，区间为半开区间 [start, end)。"""
    spans = []
    for e in ents:
        start, end = int(e["start"]), int(e["end"])
        if 0 <= start < end <= len(text):
            spans.append((start, end, e["entity_group"]))
    # 去重 + 合并重叠（简单策略：按起点排序并优先保留较长）
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    merged = []
    for s in spans:
        if not merged or s[0] >= merged[-1][1]:
            merged.append(s)
        else:
            # 有重叠：若当前更长则替换
            if (s[1]-s[0]) > (merged[-1][1]-merged[-1][0]):
                merged[-1] = s
    return merged


def make_color_map(labels: List[str]) -> Dict[str, str]:
    palette = [
        "#ffe4e1", "#e6f7ff", "#e8f5e9", "#fff7e6", "#f3e5f5",
        "#fff0f6", "#e0f7fa", "#f9fbe7", "#ede7f6", "#fff3e0"
    ]
    return {lab: col for lab, col in zip(labels, cycle(palette))}


def render_highlight_image(text: str,
                           spans: List[Tuple[int, int, str]],
                           out_path: str):
    font = load_font()
    # 将每个字符标注所属实体（若无则 None）
    char_labels = [None] * len(text)
    for s, e, lab in spans:
        for i in range(s, e):
            if 0 <= i < len(text):
                char_labels[i] = lab

    unique_labels = []
    for lab in char_labels:
        if lab and lab not in unique_labels:
            unique_labels.append(lab)
    color_map = make_color_map(unique_labels)

    # 预排版：逐字符测量，按 MAX_WIDTH 自动换行
    dummy_img = Image.new("RGB", (MAX_WIDTH, 10), "white")
    draw = ImageDraw.Draw(dummy_img)

    lines = []  # 每行是 [(char, label, (w, h)), ...]
    current_line = []
    current_width = 0

    for idx, ch in enumerate(text):
        w, h = draw.textbbox((0, 0), ch, font=font)[2:]
        if current_width + w > MAX_WIDTH and current_line:
            lines.append(current_line)
            current_line = []
            current_width = 0
        current_line.append((ch, char_labels[idx], (w, h)))
        current_width += w
    if current_line:
        lines.append(current_line)

    # 计算画布尺寸
    line_heights = [max((item[2][1] for item in line), default=FONT_SIZE) for line in lines]
    img_width = MAX_WIDTH + PADDING * 2
    text_height = sum(line_heights) + LINE_SPACING * (len(lines) - 1)
    legend_height = 0
    if unique_labels:
        legend_height = FONT_SIZE + 20
    img_height = PADDING * 2 + text_height + (legend_height if unique_labels else 0)

    # 开始绘制
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    x0, y0 = PADDING, PADDING
    y = y0
    for li, line in enumerate(lines):
        x = x0
        line_height = line_heights[li]
        for ch, lab, (w, h) in line:
            # 背景高亮
            if lab is not None:
                draw.rectangle([x, y, x + w, y + line_height], fill=color_map[lab])
            # 文字
            draw.text((x, y), ch, fill="black", font=font)
            x += w
        y += line_height + LINE_SPACING

    # 图例
    if unique_labels:
        legend_y = y + 10
        legend_x = x0
        box_size = FONT_SIZE // 1.4
        for lab in unique_labels:
            draw.rectangle([legend_x, legend_y, legend_x + box_size, legend_y + box_size],
                           fill=color_map[lab], outline="black")
            draw.text((legend_x + box_size + 8, legend_y), lab, fill="black", font=font)
            legend_x += 180  # 下一组图例位置

    img.save(out_path)
    print(f"[OK] 已保存图片：{out_path}")


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"未找到模型目录：{MODEL_DIR}")
    ents = run_ner(SENTENCE)
    # 将 pipeline 的结果转为 (start, end, label)
    spans = build_char_spans(SENTENCE, ents)
    print("识别到的实体：")
    for s, e, lab in spans:
        print(f" - [{lab}] {SENTENCE[s:e]} (start={s}, end={e})")
    render_highlight_image(SENTENCE, spans, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
