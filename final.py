import os
import json
import logging
import ast
import argparse
import random
from typing import List, Dict, Any, Tuple
import re  # 用于 extract_representative_vocabulary 中的正则处理
from transformers import TrainerCallback
from wordcloud import WordCloud

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,
    AutoModel
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import numpy as np
from seqeval.metrics import classification_report

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans  # 用于 SVD T-SNE 聚类
import seaborn as sns  # 用于 T-SNE 绘图和热力图
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger = logging.getLogger(__name__)
logger.info(f"使用的设备: {device}")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义路径
DATA_DIR = r"F:\生物文本挖掘与知识发现概论\maize_gene_data\gene_details"
BIOBERT_MODEL_DIR = r"F:\models\BioBert\biobert-v1.1"

BERT_BASE_MODEL_NAME = r"F:\models\bert-base-uncased"
OUTPUT_DIR = r"./ner_output"  # 训练结果和模型保存路径
#全局损失记录容器
GLOBAL_LOSS_RECORDS = []  # 格式: [{'model_type': str, 'ft_method': str, 'train_losses': list, 'eval_losses': list, 'epochs': list}, ...]

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# 实验参数配置
# -----------------------------------------------------------

# 训练组 (ft_method != 'none')
TRAIN_EXPERIMENTS = [
    # D 组 (BioBERT + Full Fine-tuning)
    {'model_type': 'biobert', 'ft_method': 'full'},
    # B 组 (BERT + Full Fine-tuning)
    {'model_type': 'bert', 'ft_method': 'full'},
    # F 组 (BioBERT + LoRA)
    {'model_type': 'biobert', 'ft_method': 'lora'},
    # E 组 (BERT + LoRA)
    {'model_type': 'bert', 'ft_method': 'lora'},
]

# 所有可视化组 (包含未微调的 A, C 组)
VISUALIZE_EXPERIMENTS = TRAIN_EXPERIMENTS + [
    # C 组 (BioBERT Base - 领域基础)
    {'model_type': 'biobert', 'ft_method': 'none'},
    # A 组 (BERT Base - 通用基础)
    {'model_type': 'bert', 'ft_method': 'none'},
]

# SVD 知识发现使用的模型配置 (D 组)
SVD_MODEL_CONFIG = {'model_type': 'biobert', 'ft_method': 'full'}


# ----------------------------
# 1. 数据加载与预处理
# ----------------------------

def load_and_parse_data(data_dir: str) -> List[Dict[str, Any]]:
    """读取指定目录下所有 JSON 文件，解析并合并为一个扁平列表。"""
    all_data = []
    logger.info(f"正在从目录加载数据: {data_dir}")

    total_files = len(os.listdir(data_dir))
    print(f"共加载到{total_files}个文件的数据")

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
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
                            except (ValueError, SyntaxError, TypeError) as e:
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
    """将词的标签对齐到 BioBERT 分词器的子词 (subword) 标签。"""
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
            new_labels.append(-100)
        previous_word_idx = word_idx
    return new_labels


def convert_to_token_labels(data_item: Dict[str, Any], tokenizer, label_to_id: Dict[str, int]) -> Dict[str, Any]:
    """将原始字符级标注转换为词级 BIO 标签列表，然后对齐到 BioBERT Subword。"""
    sentence = data_item['sentence']
    annotations = data_item.get('annotation', [])
    char_labels = ['O'] * len(sentence)

    # 1. 字符级标签构建
    for _, entity_type, _, span in annotations:
        if entity_type == 'None': continue
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

    # 4. 返回结果
    return {
        'sentence': sentence,
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': aligned_labels,
        'words': words,
        'annotations': str(data_item.get('annotation', []))
    }


def compute_metrics(p, id_to_label):
    """计算序列标注的 F1-Score, Precision, Recall 等指标。"""
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


def plot_loss_curve(train_losses, eval_losses, epochs, save_dir, model_type, ft_method):
    """绘制训练/评估损失曲线并保存为 PNG"""
    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    plt.plot(epochs[:len(train_losses)], train_losses, label=f'Train Loss', color='blue', marker='o', markersize=2)

    # 绘制评估损失（每个 epoch 对应一个点）
    eval_epochs = epochs[::len(epochs) // len(eval_losses)] if eval_losses else []
    if eval_losses and len(eval_epochs) == len(eval_losses):
        plt.plot(eval_epochs, eval_losses, label=f'Eval Loss', color='red', marker='s', markersize=6)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve: {model_type.upper()} + {ft_method.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_plot_path = os.path.join(save_dir, f'loss_curve_{model_type}_{ft_method}.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"损失曲线已保存到: {loss_plot_path}")


def plot_combined_loss_curve():
    """将所有实验组的训练/评估损失绘制到同一张图并保存到 ner_output 目录"""
    if not GLOBAL_LOSS_RECORDS:
        logger.warning("没有损失数据可绘制合并图，跳过。")
        return

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']  # 颜色池
    markers = ['o', 's', '^', 'D', 'v', '*']  # 标记池

    for idx, record in enumerate(GLOBAL_LOSS_RECORDS):
        model_type = record['model_type'].upper()
        ft_method = record['ft_method'].upper()
        label_prefix = f"{model_type} + {ft_method}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # 绘制训练损失（降采样避免点过于密集）
        train_losses = record['train_losses']
        train_epochs = record['epochs']
        # 每10个点取1个，保证图的清晰度
        sample_step = max(1, len(train_losses) // 200)
        sampled_train_losses = train_losses[::sample_step]
        sampled_train_epochs = train_epochs[::sample_step]

        plt.plot(
            sampled_train_epochs, sampled_train_losses,
            label=f'{label_prefix} (Train)',
            color=color, marker=marker, markersize=3, alpha=0.7, linewidth=1.5
        )

        # 绘制评估损失
        eval_losses = record['eval_losses']
        eval_epochs = record['epochs']
        if eval_losses:
            # 评估损失按epoch对应，取每个epoch的最后一个步骤作为坐标
            eval_step_interval = len(eval_epochs) // len(eval_losses)
            eval_plot_epochs = eval_epochs[eval_step_interval - 1::eval_step_interval][:len(eval_losses)]
            plt.plot(
                eval_plot_epochs, eval_losses,
                label=f'{label_prefix} (Eval)',
                color=color, marker=marker, markersize=6, alpha=1.0, linewidth=2, linestyle='--'
            )

    # 图表美化设置
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Combined Loss Curves of All NER Experiments', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存到 ner_output 目录
    combined_plot_path = os.path.join(OUTPUT_DIR, 'combined_loss_curve_all_experiments.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"合并损失曲线已保存到: {combined_plot_path}")


# ----------------------------
# 2. NER 训练与评估 (保持不变)
# ----------------------------

def run_ner_training(model_path: str, ft_method: str, train_dataset, eval_dataset, bio_labels: List[str]):
    """执行全参数微调或 LoRA 微调，并保存模型。"""

    model_name = model_path
    path_safe_name = "biobert" if 'biobert' in model_path.lower() else "bert"
    model_save_dir = os.path.join(OUTPUT_DIR, f"{path_safe_name}_{ft_method}_model")

    # 检查模型是否已存在 (简化检查)
    if os.path.exists(model_save_dir) and os.listdir(model_save_dir):
        logger.info(f"==> 模型 ({path_safe_name.upper()} + {ft_method.upper()}) 已存在于 {model_save_dir}，跳过训练步骤。")
        return

    os.makedirs(model_save_dir, exist_ok=True)

    label_to_id = {label: i for i, label in enumerate(bio_labels)}
    id_to_label = {i: label for i, label in enumerate(bio_labels)}

    logger.info(f"==> 正在加载模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(bio_labels),
        id2label=id_to_label,
        label2id=label_to_id
    ).to(device)

    if ft_method == "lora":
        logger.info("应用 LoRA (Low-Rank Adaptation) 配置...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 统计 LoRA 微调时需要训练的参数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"LoRA 微调需要训练的参数量: {trainable_params:,}")

    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"全参数微调。总可训练参数: {total_params:,}")

    # training_args = TrainingArguments(
    #     output_dir=model_save_dir,
    #     num_train_epochs=5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir='./logs',
    #     logging_steps=100,
    #     load_best_model_at_end=False,
    #     metric_for_best_model="f1",
    #     greater_is_better=True,
    #     report_to="none"
    # )

    # 修改 TrainingArguments，添加日志步骤和评估策略
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # 降低日志间隔，记录更多损失点
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        eval_strategy="epoch",  # 每个 epoch 结束评估
        save_strategy="epoch",  # 每个 epoch 保存模型
        logging_first_step=True,  # 记录第一步的损失
    )

    # 定义损失记录列表
    train_losses = []
    eval_losses = []
    epochs = []

    # 自定义训练回调，记录损失
    class LossLoggerCallback(TrainerCallback):
        # 实现必要的回调方法
        def on_init_end(self, args, state, control, **kwargs):
            pass  # 空实现，避免报错

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                train_losses.append(logs["loss"])
                epochs.append(state.global_step)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and "eval_loss" in metrics:
                eval_losses.append(metrics["eval_loss"])

    # 初始化回调实例
    loss_logger = LossLoggerCallback()

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=lambda p: compute_metrics(p, id_to_label)
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id_to_label),
        callbacks=[loss_logger]  # 添加损失回调
    )

    logger.info("==> 开始训练...")
    trainer.train()
    logger.info("==> 训练完成。")

    logger.info("==> 开始评估...")
    results = trainer.evaluate()
    logger.info(f"评估结果: {results}")

    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"==> 模型和分词器已保存到: {model_save_dir}")

    # 训练结束后绘制损失图
    plot_loss_curve(train_losses, eval_losses, epochs, model_save_dir, path_safe_name, ft_method)

    # 将当前实验组的损失数据添加到全局容器
    GLOBAL_LOSS_RECORDS.append({
        'model_type': path_safe_name,
        'ft_method': ft_method,
        'train_losses': train_losses.copy(),
        'eval_losses': eval_losses.copy(),
        'epochs': epochs.copy()
    })

    return results


# ----------------------------
# 2.5. 仅评估函数 (针对未微调的 A, C 组)
# ----------------------------

def run_evaluation_only(model_type: str, ft_method: str, eval_dataset, bio_labels: List[str]):
    """
    加载基础模型 (未微调的 A, C 组)，并对评估集进行性能评估。
    """
    path_safe_name = model_type

    # 确定模型加载路径
    if model_type == 'biobert':
        model_name = BIOBERT_MODEL_DIR
        group_name = "C-Group (BioBERT Base)"
    else:  # model_type == 'bert'
        model_name = BERT_BASE_MODEL_NAME
        group_name = "A-Group (BERT Base)"

    logger.info(f"\n--- 正在运行评估实验: {group_name} ---")

    label_to_id = {label: i for i, label in enumerate(bio_labels)}
    id_to_label = {i: label for i, label in enumerate(bio_labels)}

    try:
        # 基础模型加载，用于 TokenClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(bio_labels),
            id2label=id_to_label,
            label2id=label_to_id
        ).to(device)
    except Exception as e:
        logger.error(f"加载基础模型 {model_name} 失败: {e}。跳过评估。")
        return

    # 使用 Trainer 进行评估
    # 这里我们不需要 TrainingArguments 中的训练参数，只需使用 Trainer 的评估功能
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id_to_label)
    )

    logger.info("==> 开始评估...")
    results = trainer.evaluate(eval_dataset)

    logger.info(f"评估完成。{group_name} 结果:")
    # 将结果写入一个文件，以便于比较
    result_file = os.path.join(OUTPUT_DIR, "evaluation_results.json")

    # 尝试读取现有结果，或创建一个新的字典
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    key = f"{model_type}_{ft_method}_evaluation"
    all_results[key] = results

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    logger.info(f"评估结果已记录在 {result_file} 中。")
    return results

# ----------------------------
# 3. 嵌入可视化 (Embedding Visualization)
# ----------------------------

def extract_representative_vocabulary(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    从标注数据中提取实体词汇，并添加通用词和领域动词作为参考点。
    返回 {词汇: 标签} 字典。
    """
    entity_dict = {}

    for item in data:
        # annotation 结构是 [entity_name, entity_type, offset, span]
        for entity_name, entity_type, _, _ in item.get('annotation', []):
            if entity_type and entity_type != 'None':
                name = entity_name.strip()
                name = re.sub(r'[^\w\s-]', '', name)

                if not name: continue
                # 仅考虑单词或短语的第一个词 (简化处理)
                word = name.split()[0].lower()

                if word not in entity_dict:
                    entity_dict[word] = set()
                entity_dict[word].add(entity_type)

    word_labels = {}
    for word, types in entity_dict.items():
        if len(types) > 0:
            # 使用第一个发现的实体类型作为标签
            word_labels[word] = sorted(list(types))[0]

            # 添加参考点：通用词和领域动词
    common_words = [
        "the", "is", "was", "and", "but", "in", "to", "for",
        "structure", "analysis", "study", "showed"
    ]
    for word in common_words:
        if word.lower() not in word_labels:
            word_labels[word.lower()] = "O_Common"

    domain_verbs = [
        "activate", "inhibit", "regulates", "expressed", "binds", "mutant"
    ]
    for word in domain_verbs:
        if word.lower() not in word_labels:
            word_labels[word.lower()] = "Domain_Verb"

    logger.info(f"共提取 {len(word_labels)} 个代表性词汇进行可视化。")
    return word_labels


def get_word_embeddings(words: List[str], model_name_or_path: str, tokenizer, is_finetuned: bool,
                        word_label_map: Dict[str, str]) -> Tuple[
    np.ndarray, List[str], List[str]]:
    """
    加载模型并为词汇列表提取词嵌入向量，支持 LoRA 适配器加载。
    """
    logger.info(f"正在加载模型: {model_name_or_path} 到 {device}...")

    model = None
    try:
        model = AutoModel.from_pretrained(model_name_or_path)
    except Exception as e:
        if is_finetuned and ('lora' in model_name_or_path.lower() or 'full' not in model_name_or_path.lower()):
            base_model_path = BIOBERT_MODEL_DIR if 'biobert' in model_name_or_path.lower() else BERT_BASE_MODEL_NAME

            model = AutoModel.from_pretrained(base_model_path)

            try:
                adapter_path = model_name_or_path
                model.load_adapter(adapter_path, adapter_name="default")
                model.set_active_adapter("default")
                logger.info(f"成功加载基础模型并应用 LoRA 适配器: {adapter_path}")
            except Exception as e_lora:
                logger.error(f"加载 LoRA 适配器失败: {e_lora}. 尝试使用基础模型。")
        else:
            logger.error(f"模型加载失败，请检查路径: {model_name_or_path}. 错误: {e}")
            raise e

    model.to(device)
    model.eval()

    embeddings = []
    processed_words = []

    BATCH_SIZE = 32
    for i in range(0, len(words), BATCH_SIZE):
        batch_words = words[i:i + BATCH_SIZE]

        inputs = tokenizer(
            [" " + word + " " for word in batch_words],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        for j, word in enumerate(batch_words):
            word_lower = word.lower()
            if last_hidden_state.shape[1] > 1:
                # 提取第一个子词的嵌入（索引 1）
                word_embedding_tensor = last_hidden_state[j, 1, :]
                word_embedding = word_embedding_tensor.cpu().numpy()
                embeddings.append(word_embedding)
                processed_words.append(word_lower)
            elif last_hidden_state.shape[1] > 0:
                # 备选：使用 [CLS] 向量
                word_embedding = last_hidden_state[j, 0, :].cpu().numpy()
                embeddings.append(word_embedding)
                processed_words.append(word_lower)
            else:
                pass

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    # 使用传入的 word_label_map 来获取标签
    final_labels = [word_label_map.get(w, 'N/A_Error') for w in processed_words]

    return np.array(embeddings), processed_words, final_labels


def run_tsne_visualization(model_type: str, ft_method: str, is_finetuned: bool, word_labels: Dict[str, str]):
    """
    加载特定模型，提取动态词汇的嵌入，并进行 T-SNE 可视化。
    """
    path_safe_name = model_type
    if is_finetuned:
        model_load_path = os.path.join(OUTPUT_DIR, f"{path_safe_name}_{ft_method}_model")
    else:
        model_load_path = BIOBERT_MODEL_DIR if model_type == 'biobert' else BERT_BASE_MODEL_NAME

    if is_finetuned and not os.path.exists(model_load_path):
        logger.error(f"错误: 找不到微调后的模型文件: {model_load_path}。跳过可视化 {model_type.upper()}/{ft_method.upper()}。")
        return

    logger.info(f"\n--- 正在进行 T-SNE 可视化：{model_type.upper()} with {ft_method.upper()} ---")

    base_model_path = BIOBERT_MODEL_DIR if model_type == 'biobert' else BERT_BASE_MODEL_NAME
    tokenizer_path = model_load_path if is_finetuned and os.path.exists(model_load_path) else base_model_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"加载 Tokenizer 失败: {tokenizer_path}. 错误: {e}")
        return

    # --- 1. 准备词汇列表 ---
    words_to_plot = list(word_labels.keys())

    # --- 2. 提取嵌入 (传入 word_labels) ---
    embeddings, final_words, final_labels = get_word_embeddings(
        words_to_plot,
        model_load_path,
        tokenizer,
        is_finetuned,
        word_labels  # 传入标签映射字典
    )

    if embeddings.shape[0] < 5:
        logger.error("嵌入提取失败或数量太少，无法进行 T-SNE。")
        return

    # --- 3. 降维与可视化 ---
    logger.info(f"提取到 {embeddings.shape[0]} 个嵌入向量，开始 T-SNE 降维...")

    tsne_params = {'n_components': 2, 'random_state': SEED, 'perplexity': 30, 'max_iter': 3000}
    tsne = TSNE(**tsne_params)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. 绘图
    plt.figure(figsize=(10, 10))

    # 绘制散点图
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=final_labels,  # 使用正确的标签列表
        palette=sns.color_palette("hls", len(set(final_labels))),
        legend="full",
        alpha=0.6,
        s=30
    )

    plt.title(f"T-SNE Visualization: {model_type.upper()} with {ft_method.upper()} (Semantic Labels)")
    plt.legend(loc='best', fontsize=8)

    # 标注少量高频或关键词汇
    sample_words_indices = random.sample(range(len(final_words)), min(20, len(final_words)))
    for idx in sample_words_indices:
        plt.annotate(final_words[idx], (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), fontsize=7, alpha=0.8)

    plot_path = os.path.join(OUTPUT_DIR, f"tsne_{model_type}_{ft_method}_dynamic.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"T-SNE 图已保存到: {plot_path}")


# ----------------------------
# 4. SVD 知识发现 (Knowledge Discovery) - 核心修复部分
# ----------------------------

def get_ner_predictions(model_load_path: str, dataset: Dataset, bio_labels: List[str]):
    # ... (函数体与上一个回复中保持一致，用于预测实体)
    logger.info(f"--- 正在加载模型进行预测：{model_load_path} ---")

    id_to_label = {i: label for i, label in enumerate(bio_labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    model = AutoModelForTokenClassification.from_pretrained(model_load_path).to(device)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions, _, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)

    results = []
    for i in range(len(dataset)):
        words = dataset[i]['words']

        tokenized_sentence = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        word_ids = tokenized_sentence.word_ids(batch_index=0)

        predicted_labels = [id_to_label[p] for p in predictions[i]]

        entities = set()
        current_entity_type = None
        current_entity_text = []

        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                label = predicted_labels[token_idx]
                if word_idx >= len(words):
                    previous_word_idx = word_idx
                    continue
                word = words[word_idx]

                if label.startswith("B-"):
                    if current_entity_type:
                        entities.add((" ".join(current_entity_text), current_entity_type.split('-')[1]))

                    current_entity_type = label
                    current_entity_text = [word]

                elif label.startswith("I-") and current_entity_type and label.split('-')[1] == \
                        current_entity_type.split('-')[1]:
                    current_entity_text.append(word)

                elif label == "O" or (label.startswith(("B-", "I-")) and current_entity_type and label.split('-')[1] !=
                                      current_entity_type.split('-')[1]):
                    if current_entity_type:
                        entities.add((" ".join(current_entity_text), current_entity_type.split('-')[1]))
                        current_entity_type = None
                        current_entity_text = []

            previous_word_idx = word_idx

        if current_entity_type:
            entities.add((" ".join(current_entity_text), current_entity_type.split('-')[1]))

        results.append({
            'sentence': dataset[i]['sentence'],
            'predicted_entities': list(entities)
        })

    return results


def build_co_occurrence_matrix(ner_results: List[Dict[str, Any]]):
    # ... (函数体与上一个回复中保持一致，用于构建共现矩阵)
    all_entities = set()
    for item in ner_results:
        for entity, _ in item['predicted_entities']:
            all_entities.add(entity)

    entity_list = sorted(list(all_entities))
    entity_to_id = {entity: i for i, entity in enumerate(entity_list)}
    num_entities = len(entity_list)

    logger.info(f"发现 {num_entities} 个唯一实体，正在构建共现矩阵...")

    co_occurrence_matrix = np.zeros((num_entities, num_entities), dtype=np.int32)

    for item in ner_results:
        entities_in_sentence = [e for e, _ in item['predicted_entities']]

        for i in range(len(entities_in_sentence)):
            for j in range(i + 1, len(entities_in_sentence)):
                id_i = entity_to_id[entities_in_sentence[i]]
                id_j = entity_to_id[entities_in_sentence[j]]

                co_occurrence_matrix[id_i, id_j] += 1
                co_occurrence_matrix[id_j, id_i] += 1

    return co_occurrence_matrix, entity_list


# 绘制实体共现热力图
def plot_co_occurrence_heatmap(matrix, entity_list, output_dir):
    """将实体共现矩阵可视化为热力图，只显示 top N 个高频实体。"""

    # 计算每个实体的总共现次数（行/列和）
    entity_sums = np.sum(matrix, axis=0)

    # 选择共现次数最高的 TOP_N 个实体
    TOP_N = min(30, len(entity_list))
    top_indices = np.argsort(entity_sums)[-TOP_N:][::-1]

    # 筛选矩阵和实体列表
    top_matrix = matrix[top_indices, :][:, top_indices]
    top_entities = [entity_list[i] for i in top_indices]

    if top_matrix.size == 0:
        logger.warning("共现矩阵过小，无法绘制热力图。")
        return

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        top_matrix,
        xticklabels=top_entities,
        yticklabels=top_entities,
        cmap="Reds",
        linewidths=.5,
        linecolor='black',
        annot=False,  # 不在图上标注数字，更清晰
        fmt="d",
        cbar_kws={'label': 'Co-occurrence Count'}
    )
    plt.title(f"Top {TOP_N} Entity Co-occurrence Heatmap (D-Model Extracted)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"heatmap_entity_cooccurrence.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"实体共现热力图已保存到: {plot_path}")


def run_svd_analysis(dataset: Dataset, bio_labels: List[str]):
    d_model_path = os.path.join(OUTPUT_DIR, f"{SVD_MODEL_CONFIG['model_type']}_{SVD_MODEL_CONFIG['ft_method']}_model")

    if not os.path.exists(d_model_path):
        logger.error(f"\n--- SVD 分析失败 ---")
        logger.error(f"错误: 找不到 SVD 所需的 D 组模型 ({d_model_path})。请确保训练已成功完成。")
        return

    logger.info(f"\n==========================================")
    logger.info(f"== 启动 SVD 知识发现 (基于 D 组模型) ==")
    logger.info(f"==========================================")

    # 1. 获取 NER 预测结果和共现矩阵
    ner_results = get_ner_predictions(d_model_path, dataset, bio_labels)
    co_occurrence_matrix, entity_list = build_co_occurrence_matrix(ner_results)

    if co_occurrence_matrix.shape[0] < 50:
        logger.warning(f"实体数量过少 ({co_occurrence_matrix.shape[0]})，SVD 结果可能不具备代表性。")
        if co_occurrence_matrix.shape[0] == 0:
            logger.error("无法执行 SVD 分析。")
            return

    # 绘制热力图
    plot_co_occurrence_heatmap(co_occurrence_matrix, entity_list, OUTPUT_DIR)

    # 2. SVD/LSA 分解
    k = min(50, co_occurrence_matrix.shape[0] - 1)
    logger.info(f"对 {co_occurrence_matrix.shape[0]}x{co_occurrence_matrix.shape[0]} 矩阵进行 SVD 分解，提取 {k} 个主题...")
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    entity_vectors = svd.fit_transform(co_occurrence_matrix)

    logger.info("\n--- 潜在主题分析 (SVD) ---")
    # ... (主题分析代码保持不变)
    num_topics_to_show = 5
    topic_components = svd.components_

    for topic_idx, topic in enumerate(topic_components[:num_topics_to_show]):
        top_entities_indices = topic.argsort()[:-6:-1]
        top_entities = [entity_list[i] for i in top_entities_indices]

        print(f"主题 {topic_idx + 1} (方差解释率: {svd.explained_variance_ratio_[topic_idx]:.4f}):")
        print(f"  核心实体: {top_entities}")

    # 3. 可视化 (T-SNE + K-Means 聚类)
    logger.info("将 SVD 实体向量进行 T-SNE 可视化...")

    # 对 SVD 向量进行 T-SNE 聚类
    tsne_svd = TSNE(n_components=2, random_state=SEED, perplexity=min(30, co_occurrence_matrix.shape[0] - 1),
                    metric='cosine')
    entity_vectors_2d = tsne_svd.fit_transform(entity_vectors)

    # 使用 K-Means 聚类来生成颜色标签 (假设有 5 个主要潜在主题)
    N_CLUSTERS = min(10, co_occurrence_matrix.shape[0] // 5)  # 聚类数量
    if N_CLUSTERS < 2:
        logger.warning("实体数量不足，无法进行有效聚类。T-SNE 将不显示颜色。")
        cluster_labels = ['Cluster 1'] * entity_vectors_2d.shape[0]
        unique_labels = cluster_labels
    else:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
        cluster_labels_id = kmeans.fit_predict(entity_vectors_2d)
        cluster_labels = [f"Topic Cluster {i + 1}" for i in cluster_labels_id]
        unique_labels = sorted(list(set(cluster_labels)))

    plt.figure(figsize=(10, 10))

    # 使用 seaborn 绘制，利用 cluster_labels 作为颜色/图注
    sns.scatterplot(
        x=entity_vectors_2d[:, 0],
        y=entity_vectors_2d[:, 1],
        hue=cluster_labels,  # 使用聚类结果作为颜色标签
        palette=sns.color_palette("tab10", len(unique_labels)),
        legend="full",
        alpha=0.6,
        s=30
    )

    plt.title(
        f"T-SNE Visualization of SVD Entity-Topic Vectors (D-Model Extracted) with K-Means Clustering ({len(unique_labels)} Topics)")
    plt.legend(loc='best', fontsize=8, title="Potential Topics")

    # 标注部分高频实体
    # if co_occurrence_matrix.shape[0] > 0:
    #     high_freq_indices = [i for i, e in enumerate(entity_list) if co_occurrence_matrix[i, i] > 5]
    #     # 标注 20 个随机高频实体
    #     for idx in random.sample(high_freq_indices, min(20, len(high_freq_indices))):
    #         plt.annotate(entity_list[idx], (entity_vectors_2d[idx, 0], entity_vectors_2d[idx, 1]), fontsize=7)

    # 标注部分实体（按概率从所有实体中随机标记，避免过度拥挤）
    if co_occurrence_matrix.shape[0] > 0:
        # 1. 直接获取所有实体的索引（跳过高频筛选，所有实体均为候选）
        all_entity_indices = list(range(len(entity_list)))  # 所有实体的索引列表
        if not all_entity_indices:
            logger.warning("无实体可标注，跳过标注步骤。")
        else:
            # 2. 配置标注参数：概率（30%概率标记）、最大标注数量（避免图拥挤）
            annotation_prob = 0.3  # 自定义：30%的概率标记一个实体（可按需调整）
            max_annotations = 80  # 自定义：最多标注80个实体（避免标注过多遮挡）
            selected_indices = []

            # 3. 按概率筛选实体（遍历所有实体，按概率选中，直到达最大数量）
            for idx in all_entity_indices:
                if len(selected_indices) >= max_annotations:
                    break  # 达到最大标注数量，停止筛选
                # 随机生成0-1的概率，小于设定值则选中该实体
                if random.random() < annotation_prob:
                    selected_indices.append(idx)

            # 4. 对选中的实体进行标注（标在点旁，保持原有美化格式）
            logger.info(f"按 {annotation_prob * 100}% 概率从所有实体中选中 {len(selected_indices)} 个实体进行标注")
            for idx in selected_indices:
                entity_name = entity_list[idx]
                # 标注位置微调（x/y轴偏移0.1，避免与点重叠）
                plt.annotate(
                    entity_name,
                    xy=(entity_vectors_2d[idx, 0], entity_vectors_2d[idx, 1]),
                    xytext=(0.1, 0.1),  # 文本相对点的偏移量（单位：图坐标）
                    textcoords="offset points",
                    fontsize=6,  # 字体大小，避免遮挡其他元素
                    alpha=0.8,  # 透明度，提升可读性
                )

    plot_path = os.path.join(OUTPUT_DIR, f"tsne_svd_entity_cooccurrence.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"SVD 向量 T-SNE 图已保存到: {plot_path}")


def generate_ground_truth_wordcloud(data: List[Dict[str, Any]], output_dir: str):
    """
    从原始数据中提取真实标注的实体，生成词云图。
    """
    all_entities = []
    for item in data:
        for annotation in item.get('annotation', []):
            if len(annotation) >= 1:
                entity_name = annotation[0].strip()
                if entity_name:
                    all_entities.append(entity_name)

    if not all_entities:
        logger.warning("没有提取到真实标注的实体，跳过词云生成。")
        return

    # 将实体列表转换为字符串，用空格分隔
    text = ' '.join(all_entities)

    # 生成词云
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)

    # 保存词云图
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Ground Truth Entities Word Cloud')

    wordcloud_path = os.path.join(output_dir, 'ground_truth_entities_wordcloud.png')
    plt.savefig(wordcloud_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"真实标注实体的词云已保存到: {wordcloud_path}")

# ----------------------------
# 5. 主程序
# ----------------------------

def main():
    # 检查路径
    if not os.path.exists(BIOBERT_MODEL_DIR):
        logger.error(f"错误：找不到 BioBERT 模型文件。请确保路径 {BIOBERT_MODEL_DIR} 正确。")
        return
    if not os.path.exists(DATA_DIR):
        logger.error(f"错误：找不到数据文件。请确保路径 {DATA_DIR} 正确。")
        return

    # 1. 数据加载与标签体系构建
    raw_data = load_and_parse_data(DATA_DIR)
    if not raw_data: return

    entity_types = extract_unique_entity_types(raw_data)
    if not entity_types: return

    bio_labels = build_bio_labels(entity_types)
    label_to_id = {label: i for i, label in enumerate(bio_labels)}
    logger.info(f"构建的 BIO 标签集 (包含 {len(entity_types)} 种实体): {bio_labels}")

    # 提取代表性词汇用于 T-SNE 可视化
    word_labels_for_tsne = extract_representative_vocabulary(raw_data)

    # 2. 数据预处理
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_DIR)
    processed_data = []
    for item in raw_data:
        try:
            processed_item = convert_to_token_labels(item, tokenizer, label_to_id)
            processed_data.append(processed_item)
        except Exception as e:
            logger.warning(f"转换句子失败: {item.get('sentence', 'N/A')[:50]}... 错误: {e}")

    # 定义 HuggingFace Dataset 的 features
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
    all_data_dataset = dataset

    # ==========================================================
    # 步骤 1: NER 训练与评估 (运行 B, D, E, F 组)
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 1. NER 训练与评估 (B, D, E, F 组) ==")
    logger.info("==========================================")

    for config in TRAIN_EXPERIMENTS:
        model_type = config['model_type']
        ft_method = config['ft_method']

        model_load_path = BIOBERT_MODEL_DIR if model_type == 'biobert' else BERT_BASE_MODEL_NAME

        logger.info(f"\n--- 正在运行训练实验: {model_type.upper()} + {ft_method.upper()} ---")
        run_ner_training(model_load_path, ft_method, train_dataset, eval_dataset, bio_labels)

    # ==========================================================
    # 步骤 1.5: 评估未微调模型 (A, C 组)
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 1.5. 仅评估未微调的基础模型 (A, C 组) ==")
    logger.info("==========================================")

    # A 组: BERT Base (ft_method='none')
    run_evaluation_only('bert', 'none', eval_dataset, bio_labels)

    # C 组: BioBERT Base (ft_method='none')
    run_evaluation_only('biobert', 'none', eval_dataset, bio_labels)

    # ==========================================================
    # 步骤 2: 嵌入可视化 (运行 A, B, C, D, E, F 组)
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 2. 嵌入可视化 (A, B, C, D, E, F 组) ==")
    logger.info("==========================================")

    for config in VISUALIZE_EXPERIMENTS:
        model_type = config['model_type']
        ft_method = config['ft_method']
        is_finetuned = (ft_method != 'none')

        run_tsne_visualization(model_type, ft_method, is_finetuned, word_labels_for_tsne)



    # ==========================================================
    # 步骤 3: SVD 知识发现 (使用 D 组模型)
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 3. SVD 知识发现 (基于 D 组模型) ==")
    logger.info("==========================================")

    run_svd_analysis(all_data_dataset, bio_labels)

    # ==========================================================
    # 步骤 4: 绘制所有实验组的合并损失曲线
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 4. 绘制合并损失曲线 ==")
    logger.info("==========================================")
    plot_combined_loss_curve()

    # ==========================================================
    # 步骤 5: 生成真实标注实体的词云
    # ==========================================================
    logger.info("\n==========================================")
    logger.info("== 启动 5. 生成真实标注实体的词云 ==")
    logger.info("==========================================")
    generate_ground_truth_wordcloud(raw_data, OUTPUT_DIR)

    logger.info("\n==========================================")
    logger.info("== 所有实验步骤执行完毕 ==")
    logger.info("==========================================")


if __name__ == "__main__":
    main()