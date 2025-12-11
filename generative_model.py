import os
import json
import logging
import requests
import time
import re
import ast
import random
import numpy as np
from typing import List, Set, Dict, Any

from datasets import Dataset

# ===========================
# 全局配置
# ===========================
# 1. 保持与 Prompt Tuning 代码完全一致的种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = r"F:\生物文本挖掘与知识发现概论\maize_gene_data\gene_details"
KEY_FILE_PATH = r"F:\silicon_flow_key.txt"
OUTPUT_FILE = r"./llm_ner_full_test_results.json"


# API_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
# API_MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"
# API_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1-Terminus"
# API_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"
API_MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ===========================
# 数据处理函数 (复用 Prompt Tuning 逻辑)
# ===========================

def load_and_parse_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    复用 Prompt Tuning 的数据加载逻辑
    """
    all_data = []
    logger.info(f"正在从目录加载数据: {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"目录不存在: {data_dir}")
        return []

    file_list = os.listdir(data_dir)
    # 为了保证顺序一致性，最好排序一下
    file_list.sort()

    for filename in file_list:
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = json.load(f)

                if not isinstance(file_content, list):
                    file_content = [file_content] if isinstance(file_content, dict) else []

                for item in file_content:
                    if not isinstance(item, dict) or ("sentence" not in item) or ("annotation" not in item):
                        continue

                    # 解析 annotation
                    annotation_data = item['annotation']
                    if isinstance(annotation_data, str):
                        try:
                            item['annotation'] = ast.literal_eval(annotation_data)
                        except:
                            item['annotation'] = []

                    if not isinstance(item['annotation'], (list, tuple)):
                        item['annotation'] = []

                    all_data.append(item)

        except Exception as e:
            pass  # 忽略错误文件，与原逻辑保持一致

    logger.info(f"总共加载了 {len(all_data)} 条有效记录。")
    return all_data


def prepare_test_dataset(all_data: List[Dict[str, Any]]):
    """
    使用 datasets 库和相同的种子进行切分
    """
    # 提取需要的字段
    processed_data = []
    for item in all_data:
        sentence = item['sentence']
        annotations = item.get('annotation', [])

        # 提取 Ground Truth (真实标签)
        gt_entities = set()
        for anno in annotations:
            # annotation 格式通常是 [ID, Type, Start, End, "EntityName"]
            # 遍历寻找字符串字段作为实体名
            for field in anno:
                if isinstance(field, str) and field in sentence and len(field) > 1 and field != 'Gene':
                    gt_entities.add(field.strip())

        # 即使没有实体的句子也应该保留，因为 False Positive 也是一种错误
        processed_data.append({
            "sentence": sentence,
            "ground_truth": list(gt_entities)
        })

    # 创建 Dataset 对象
    dataset = Dataset.from_list(processed_data)

    # 【关键】使用与 Prompt Tuning 完全一致的 split 参数
    train_test_split = dataset.train_test_split(test_size=0.1, seed=SEED)
    test_dataset = train_test_split['test']

    logger.info(f"数据集划分完成。测试集大小: {len(test_dataset)} (Train: {len(train_test_split['train'])})")
    return test_dataset


# ===========================
# LLM 调用函数
# ===========================

def read_api_key(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readline().strip()
    except Exception as e:
        logger.error(f"读取 Key 失败: {e}")
        return ""


def extract_entities_with_llm(sentence: str, api_key: str) -> List[str]:
    """调用 DeepSeek-V3 进行 NER 提取"""

    prompt = f"""
    你是一个生物信息学专家。请从下面的文本中提取所有的生物医学实体”。

    文本："{sentence}"

    输出要求：
    1. 仅输出一个 JSON 列表。
    2. 列表包含提取到的基因名称字符串。
    3. 如果没有找到基因，输出空列表 []。
    4. 不要包含任何其他解释或 Markdown 标记。
    """

    payload = {
        "model": API_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 4096,
        "enable_thinking": False,
        "thinking_budget": 4096,
        "min_p": 0.05,
        "temperature": 0.0,  # 设为0以保证评估的可复现性
        "top_p": 0.7,
        "frequency_penalty": 0.0,
        "response_format": {"type": "text"},
        "tools": []
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.siliconflow.cn/v1/chat/completions",
                                     json=payload, headers=headers, timeout=60)

            if response.status_code == 200:
                res_json = response.json()
                if "choices" in res_json:
                    content = res_json["choices"][0]["message"]["content"]
                    content = content.replace("```json", "").replace("```", "").strip()
                    if "[" in content and "]" in content:
                        match = re.search(r'\[.*\]', content, re.DOTALL)
                        if match: content = match.group(0)

                    try:
                        entities = json.loads(content)
                        if isinstance(entities, list): return entities
                        if isinstance(entities, dict): return list(entities.values())[0] if entities else []
                    except:
                        pass  # JSON解析失败，继续重试或返回空
                return []  # 格式不对返回空

            elif response.status_code == 429:  # Rate limit
                time.sleep(2)
                continue
            else:
                logger.warning(f"API Error {response.status_code}")
                return []

        except Exception as e:
            time.sleep(1)

    return []


# ===========================
# 评估指标计算
# ===========================

def calculate_metrics(results):
    """
    计算整个数据集的 P/R/F1
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for item in results:
        gt = set(item['ground_truth'])
        pred = set(item['prediction'])

        # 标准化处理 (忽略大小写和首尾空格)
        gt_norm = {x.lower().strip() for x in gt}
        pred_norm = {x.lower().strip() for x in pred}

        tp = len(gt_norm.intersection(pred_norm))
        fp = len(pred_norm - gt_norm)
        fn = len(gt_norm - pred_norm)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1


# ===========================
# 主程序
# ===========================

def main():
    api_key = read_api_key(KEY_FILE_PATH)
    if not api_key: return

    # 1. 加载全量数据
    all_raw_data = load_and_parse_data(DATA_DIR)
    if not all_raw_data:
        return

    # 2. 准备测试集 (使用与 Prompt Tuning 相同的 Split 逻辑)
    # 这确保了 Prompt Tuning 评估的 test set 和这里用的 test set 是同一批数据
    test_dataset = prepare_test_dataset(all_raw_data)

    results = []
    total_count = len(test_dataset)

    logger.info(f"==> 开始在测试集上评估生成式模型 ({API_MODEL_NAME})")
    logger.info(f"==> 样本总数: {total_count}")

    # 3. 遍历测试集进行推理
    start_time = time.time()
    for idx, item in enumerate(test_dataset):
        sentence = item['sentence']
        ground_truth = item['ground_truth']

        prediction = extract_entities_with_llm(sentence, api_key)

        results.append({
            "sentence": sentence,
            "ground_truth": ground_truth,
            "prediction": prediction
        })

        # 进度打印
        if (idx + 1) % 10 == 0 or (idx + 1) == total_count:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remain_time = avg_time * (total_count - idx - 1)
            print(f"\r进度: [{idx + 1}/{total_count}] | 耗时: {elapsed:.1f}s | 剩余: {remain_time:.1f}s", end="")

            # 实时保存，防止中断
            if (idx + 1) % 50 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n")  
    logger.info("推理完成，开始计算最终指标...")

    # 4. 计算并输出最终指标
    p, r, f1 = calculate_metrics(results)

    print("\n" + "=" * 50)
    print(f"最终测试集评估结果 (一致性数据划分):")
    print(f"Dataset Size: {total_count}")
    print(f"Model:        {API_MODEL_NAME}")
    print("-" * 50)
    print(f"Precision:    {p:.4f}")
    print(f"Recall:       {r:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("=" * 50 + "\n")

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()