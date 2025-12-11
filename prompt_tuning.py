import os
import json
import logging
import ast
import random
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import (
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import numpy as np
from seqeval.metrics import classification_report

import torch

import matplotlib
matplotlib.use("Agg")  # 防止在无显示环境报错
import matplotlib.pyplot as plt

# ===========================
# 全局配置
# ===========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"使用的设备: {device}")

DATA_DIR = r"F:\生物文本挖掘与知识发现概论\maize_gene_data\gene_details"
BIOBERT_MODEL_DIR = r"F:\models\BioBert\biobert-v1.1"
BERT_BASE_MODEL_NAME = r"F:\models\bert-base-uncased"
OUTPUT_DIR = r"./ner_output_prompt_only"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompt 的虚拟 token 数列表：从 10 到 60（步长 10）
PROMPT_LENGTH_LIST = [10, 20, 30, 40, 50, 60]

# 只保留 Prompt Tuning 的两组实验
TRAIN_EXPERIMENTS = [
    {'model_type': 'biobert', 'ft_method': 'prompt'},  # BioBERT + Prompt
    {'model_type': 'bert',   'ft_method': 'prompt'},   # BERT   + Prompt
]


# ===========================
# 1. 数据加载与预处理
# ===========================

def load_and_parse_data(data_dir: str) -> List[Dict[str, Any]]:
    """读取指定目录下所有 JSON 文件，解析并合并为一个扁平列表。"""
    all_data = []
    logger.info(f"正在从目录加载数据: {data_dir}")

    total_files = len(os.listdir(data_dir))
    print(f"共加载到{total_files}个文件的数据")

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = json.load(f)

                if not isinstance(file_content, list):
                    if isinstance(file_content, dict):
                        logger.warning(f"文件 {filename} 的顶级结构不是列表，尝试将其视为单条记录。")
                    file_content = [file_content] if isinstance(file_content, dict) else []

                for item in file_content:
                    if not isinstance(item, dict) or ("sentence" not in item) or ("annotation" not in item):
                        continue

                    annotation_data = item['annotation']
                    if isinstance(annotation_data, str):
                        try:
                            item['annotation'] = ast.literal_eval(annotation_data)
                        except (ValueError, SyntaxError, TypeError):
                            item['annotation'] = []

                    if not isinstance(item['annotation'], (list, tuple)):
                        item['annotation'] = []

                    all_data.append(item)

        except json.JSONDecodeError:
            logger.error(f"文件 {filename} 不是有效的 JSON 格式，跳过。")
        except Exception as e:
            logger.error(f"处理文件 {filename} 遇到未知错误: {e}")

    logger.info(f"总共加载了 {len(all_data)} 条有效记录。")
    return all_data


def extract_unique_entity_types(data: List[Dict[str, Any]]) -> List[str]:
    """从标注数据中提取所有唯一的实体类型。"""
    entity_types = set()
    for item in data:
        for annotation_item in item.get('annotation', []):
            if isinstance(annotation_item, (list, tuple)) and len(annotation_item) >= 2:
                entity_type = annotation_item[1]
                if entity_type and entity_type != 'None':
                    entity_types.add(entity_type)
    return sorted(list(entity_types))


def build_bio_labels(entity_types: List[str]) -> List[str]:
    """根据实体类型构建标准的 B-I-O 标签体系。"""
    bio_labels = ["O"]
    for entity_type in entity_types:
        bio_labels.append(f"B-{entity_type}")
        bio_labels.append(f"I-{entity_type}")
    return bio_labels


def align_labels_with_tokens(word_ids: List[int], original_labels: List[int]) -> List[int]:
    """将词的标签对齐到子词 (subword) 标签。"""
    new_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:
            if word_idx < len(original_labels):
                new_labels.append(original_labels[word_idx])
            else:
                new_labels.append(-100)
        else:
            # 同一个 word 分裂出的 subword 只标首个，其余置 -100
            new_labels.append(-100)
        previous_word_idx = word_idx
    return new_labels


def convert_to_token_labels(
    data_item: Dict[str, Any],
    tokenizer,
    label_to_id: Dict[str, int]
) -> Dict[str, Any]:
    """将原始字符级标注转换为词级 BIO 标签，然后对齐到 subword。"""
    sentence = data_item['sentence']
    annotations = data_item.get('annotation', [])
    char_labels = ['O'] * len(sentence)

    # 1. 字符级标签构建
    for _, entity_type, _, span in annotations:
        if entity_type == 'None':
            continue
        if isinstance(span, (list, tuple)) and len(span) == 2:
            start_offset, end_offset = span
        else:
            continue
        if start_offset < len(sentence) and end_offset <= len(sentence) and start_offset < end_offset:
            char_labels[start_offset] = f"B-{entity_type}"
            for i in range(start_offset + 1, end_offset):
                if i < len(sentence):
                    char_labels[i] = f"I-{entity_type}"

    # 2. 词级标签构建
    words = sentence.split()
    word_labels = []
    current_char_index = 0
    for word in words:
        word_start_index = sentence.find(word, current_char_index)
        if word_start_index != -1:
            word_label = char_labels[word_start_index]
            word_labels.append(word_label)
            current_char_index = word_start_index + len(word)
            while current_char_index < len(sentence) and sentence[current_char_index] == ' ':
                current_char_index += 1
        else:
            word_labels.append('O')
            current_char_index += len(word)
            while current_char_index < len(sentence) and sentence[current_char_index] == ' ':
                current_char_index += 1

    # 3. Tokenizer 分词与标签对齐
    tokenized_input = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    word_ids = tokenized_input.word_ids(batch_index=0)
    word_label_ids = [label_to_id.get(label, label_to_id['O']) for label in word_labels]
    aligned_labels = align_labels_with_tokens(word_ids, word_label_ids)

    return {
        'sentence': sentence,
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': aligned_labels,
        'words': words,
        'annotations': str(data_item.get('annotation', []))
    }


def compute_metrics(p, id_to_label):
    """计算 P/R/F1/Accuracy（micro avg）"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    try:
        report = classification_report(true_labels, true_predictions, output_dict=True)
    except Exception as e:
        logger.warning(f"seqeval 计算失败: {e}. 返回零指标。")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    micro_avg = report.get('micro avg', {})

    metrics = {
        "precision": micro_avg.get('precision', 0.0),
        "recall": micro_avg.get('recall', 0.0),
        "f1": micro_avg.get('f1-score', 0.0),
        "accuracy": report.get('accuracy', 0.0),
    }

    return metrics


# ===========================
# 专门给 Prompt Tuning 用的 DataCollator
# ===========================

class PromptDataCollator:
    """
    在每个样本的 labels 前面加 num_virtual_tokens 个 -100，
    用来匹配 Prompt Tuning 增加的虚拟 prompt token 长度。
    """
    def __init__(self, num_virtual_tokens: int):
        self.num_virtual_tokens = num_virtual_tokens

    def __call__(self, features):
        for f in features:
            f["labels"] = [-100] * self.num_virtual_tokens + list(f["labels"])
        batch = default_data_collator(features)
        return batch


# ===========================
# 2. Prompt Tuning 训练与评估
# ===========================

def _normalize_eval_results(raw_results: Dict[str, float]) -> Dict[str, float]:
    """
    把 Trainer.evaluate() 返回的带 eval_ 前缀的结果，
    统一转换成 {precision, recall, f1, accuracy, loss}。
    """
    return {
        "precision": raw_results.get("eval_precision", 0.0),
        "recall": raw_results.get("eval_recall", 0.0),
        "f1": raw_results.get("eval_f1", 0.0),
        "accuracy": raw_results.get("eval_accuracy", 0.0),
        "loss": raw_results.get("eval_loss", 0.0),
    }


def run_ner_prompt_tuning(
    model_path: str,
    model_type: str,
    train_dataset,
    eval_dataset,
    bio_labels: List[str],
    num_virtual_tokens: int,
) -> Dict[str, float]:
    """
    只做 Prompt Tuning：
      - 冻结 backbone
      - 只训练虚拟 prompt token
      - 返回 {precision, recall, f1, accuracy, loss}
    """
    ft_method = "prompt"
    path_safe_name = "biobert" if 'biobert' in model_path.lower() else "bert"
    # 不同 prompt 长度分别存目录
    model_save_dir = os.path.join(
        OUTPUT_DIR,
        f"{path_safe_name}_{ft_method}_{num_virtual_tokens}"
    )

    label_to_id = {label: i for i, label in enumerate(bio_labels)}
    id_to_label = {i: label for i, label in enumerate(bio_labels)}

    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(f"==> 正在加载基础模型: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(bio_labels),
        id2label=id_to_label,
        label2id=label_to_id
    ).to(device)

    # 1. 冻结 backbone 参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 配置 Prompt Tuning
    prompt_config = PromptTuningConfig(
        task_type=TaskType.TOKEN_CLS,
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=model_path
    )

    # 3. 包装成 PEFT 模型
    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Prompt Tuning (virtual_tokens={num_virtual_tokens}) "
        f"需要训练的参数量: {trainable_params:,}"
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",    # 你的 transformers 版本使用这个参数名
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
    )

    data_collator = PromptDataCollator(num_virtual_tokens=num_virtual_tokens)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id_to_label)
    )

    logger.info(
        f"==> 开始 Prompt Tuning 训练: {path_safe_name.upper()} "
        f"(virtual_tokens={num_virtual_tokens})..."
    )
    trainer.train()
    logger.info("==> 训练完成。开始最终评估...")

    raw_results = trainer.evaluate()
    metrics = _normalize_eval_results(raw_results)
    logger.info(
        f"[最终评估] {path_safe_name.upper()} + PROMPT "
        f"(virtual_tokens={num_virtual_tokens}) 结果: {metrics}"
    )

    # 保存模型与 tokenizer（注意：以后如果要重新加载，需要自己写 PeftModel 的加载逻辑）
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"==> Prompt Tuning 模型和分词器已保存到: {model_save_dir}")

    # 记录结果到 json 方便后续对比
    result_file = os.path.join(OUTPUT_DIR, "prompt_tuning_results.json")
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    key = f"{path_safe_name}_prompt_{num_virtual_tokens}"
    all_results[key] = metrics

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    logger.info(f"评估结果已记录在 {result_file} 中。")
    return metrics


# ===========================
# 3. 主程序
# ===========================

def main():
    # 路径检查
    if not os.path.exists(BIOBERT_MODEL_DIR):
        logger.error(f"错误：找不到 BioBERT 模型文件。请确保路径 {BIOBERT_MODEL_DIR} 正确。")
        return
    if not os.path.exists(DATA_DIR):
        logger.error(f"错误：找不到数据文件。请确保路径 {DATA_DIR} 正确。")
        return

    # 1. 数据加载 & 标签构建
    raw_data = load_and_parse_data(DATA_DIR)
    if not raw_data:
        logger.error("数据为空，终止。")
        return

    entity_types = extract_unique_entity_types(raw_data)
    if not entity_types:
        logger.error("没有从数据中提取到实体类型，终止。")
        return

    bio_labels = build_bio_labels(entity_types)
    label_to_id = {label: i for i, label in enumerate(bio_labels)}
    logger.info(f"构建的 BIO 标签集 (包含 {len(entity_types)} 种实体): {bio_labels}")

    # 2. 预处理为 HF Dataset
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_DIR)
    processed_data = []
    for item in raw_data:
        try:
            processed_item = convert_to_token_labels(item, tokenizer, label_to_id)
            processed_data.append(processed_item)
        except Exception as e:
            logger.warning(f"转换句子失败: {item.get('sentence', 'N/A')[:50]}... 错误: {e}")

    features = Features({
        'sentence': Value(dtype='string'),
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'attention_mask': Sequence(feature=Value(dtype='int32')),
        'labels': Sequence(feature=ClassLabel(names=bio_labels)),
        'words': Sequence(feature=Value(dtype='string')),
        'annotations': Value(dtype='string')
    })

    dataset = Dataset.from_list(processed_data, features=features)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # 3. 运行两个 Prompt Tuning 实验：BioBERT + BERT
    logger.info("\n==========================================")
    logger.info("== 启动 Prompt Tuning NER 实验 (BioBERT & BERT) ==")
    logger.info("==========================================")

    # all_results[model_type][prompt_len] = metrics
    all_results: Dict[str, Dict[int, Dict[str, float]]] = {
        "biobert": {},
        "bert": {},
    }

    for config in TRAIN_EXPERIMENTS:
        model_type = config['model_type']
        ft_method = config['ft_method']  # 实际上只会是 "prompt"

        model_load_path = BIOBERT_MODEL_DIR if model_type == 'biobert' else BERT_BASE_MODEL_NAME

        for prompt_len in PROMPT_LENGTH_LIST:
            logger.info(f"\n--- 实验：{model_type.upper()} + {ft_method.upper()} + prompt_len={prompt_len} ---")
            metrics = run_ner_prompt_tuning(
                model_path=model_load_path,
                model_type=model_type,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                bio_labels=bio_labels,
                num_virtual_tokens=prompt_len,
            )
            all_results[model_type][prompt_len] = metrics

    # 4. 结果表格打印
    logger.info("\n=========== Prompt Length Sweep 结果汇总（BioBERT）===========")
    logger.info("prompt_len\tprecision\trecall\tf1\taccuracy\tloss")
    for prompt_len in PROMPT_LENGTH_LIST:
        res = all_results["biobert"][prompt_len]
        logger.info(
            f"{prompt_len}\t\t"
            f"{res['precision']:.4f}\t\t"
            f"{res['recall']:.4f}\t"
            f"{res['f1']:.4f}\t"
            f"{res['accuracy']:.4f}\t"
            f"{res['loss']:.4f}"
        )

    logger.info("\n=========== Prompt Length Sweep 结果汇总（BERT）===========")
    logger.info("prompt_len\tprecision\trecall\tf1\taccuracy\tloss")
    for prompt_len in PROMPT_LENGTH_LIST:
        res = all_results["bert"][prompt_len]
        logger.info(
            f"{prompt_len}\t\t"
            f"{res['precision']:.4f}\t\t"
            f"{res['recall']:.4f}\t"
            f"{res['f1']:.4f}\t"
            f"{res['accuracy']:.4f}\t"
            f"{res['loss']:.4f}"
        )

    # 5. 只画一张图：横轴 prompt_len，纵轴 F1，两条折线（BioBERT, BERT）
    biobert_f1 = [all_results["biobert"][pl]["f1"] for pl in PROMPT_LENGTH_LIST]
    bert_f1 = [all_results["bert"][pl]["f1"] for pl in PROMPT_LENGTH_LIST]

    plt.figure(figsize=(6, 4))
    plt.plot(PROMPT_LENGTH_LIST, biobert_f1, marker="o", label="BioBERT")
    plt.plot(PROMPT_LENGTH_LIST, bert_f1, marker="s", label="BERT")
    plt.xlabel("Number of Virtual Prompt Tokens")
    plt.ylabel("F1-score")
    plt.title("F1 vs Prompt Length (BioBERT & BERT)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = os.path.join(OUTPUT_DIR, "f1_vs_prompt_biobert_bert.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"F1-PromptLength 曲线已保存到: {save_path}")

    logger.info("======================================================")


if __name__ == "__main__":
    main()
