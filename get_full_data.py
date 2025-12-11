import requests
import json
import os
import time
import random
from typing import List, Optional

# 基础URL（项目数据接口的根地址）
BASE_URL = "http://lit-evi.hzau.edu.cn/MaizeAlterome"
# 配置参数
SAVE_DIR = "maize_gene_data"  # 数据保存总目录
RETRY_TIMES = 2  # 单个基因下载失败后的重试次数
DELAY_RANGE = (1, 2)  # 每次请求后的休眠时间（1-2秒，随机避免固定间隔）
ERROR_LOG_FILE = os.path.join(SAVE_DIR, "download_failures.log")  # 失败日志路径
# Windows系统文件名禁止的非法字符（统一替换为下划线）
ILLEGAL_CHARS = r'[\\/:*?"<>|]'


def clean_filename(filename: str) -> str:
    """
    清理文件名中的非法字符（适配Windows/Linux系统）
    :param filename: 原始文件名
    :return: 清理后的合法文件名
    """
    import re
    # 将所有非法字符替换为下划线，多个连续下划线合并为一个
    cleaned = re.sub(ILLEGAL_CHARS, '_', filename)
    cleaned = re.sub(r'_+', '_', cleaned)
    # 移除文件名开头/结尾的下划线
    cleaned = cleaned.strip('_')
    # 避免文件名过长（Windows最大路径260字符，此处限制文件名80字符）
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "..."
    return cleaned


def download_all_genes(save_dir: str) -> Optional[List[str]]:
    """
    下载所有基因列表，并提取去重后的基因名
    :param save_dir: 基因列表保存目录
    :return: 去重后的基因名列表（如["PPDK", "tb1"]），失败返回None
    """
    url = f"{BASE_URL}/all-genes/"
    try:
        print("正在获取所有基因列表...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        raw_data = response.json()  # 原始数据格式：[{"gene": "基因名1"}, {"gene": "基因名2"}, ...]

        # 提取基因名并去重（避免重复下载）
        gene_list = list({item["gene"].strip() for item in raw_data if "gene" in item and item["gene"].strip()})
        gene_list = [gene for gene in gene_list if gene]  # 过滤空字符串

        # 保存原始基因列表（可选，便于核对）
        os.makedirs(save_dir, exist_ok=True)
        raw_save_path = os.path.join(save_dir, "all_genes_raw.json")
        with open(raw_save_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        print(f"基因列表获取成功！共 {len(gene_list)} 个唯一基因")
        print(f"原始基因列表已保存至：{raw_save_path}")
        return gene_list

    except requests.exceptions.RequestException as e:
        print(f"获取基因列表失败：{e}")
        return None


def download_single_gene(gene_name: str, save_dir: str) -> bool:
    """
    下载单个基因的详细数据（带重试机制，修复文件名特殊字符问题）
    :param gene_name: 目标基因名
    :param save_dir: 单个基因数据的保存目录
    :return: 下载成功返回True，失败返回False
    """
    url = f"{BASE_URL}/searchbygene/"
    params = {"gene": gene_name}

    # 清理基因名中的非法字符，生成合法文件名
    cleaned_gene_name = clean_filename(gene_name)
    save_path = os.path.join(save_dir, f"gene_{cleaned_gene_name}.json")

    # 若文件已存在，跳过下载
    if os.path.exists(save_path):
        print(f"基因 [{gene_name}] 数据已存在，跳过下载")
        return True

    # 重试逻辑
    for retry in range(RETRY_TIMES + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            gene_data = response.json()

            # 保存数据
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(gene_data, f, ensure_ascii=False, indent=2)

            # 随机休眠，避免限流
            time.sleep(random.uniform(*DELAY_RANGE))
            print(f"基因 [{gene_name}] 下载成功（保存为：gene_{cleaned_gene_name}.json）")
            return True

        except requests.exceptions.RequestException as e:
            if retry < RETRY_TIMES:
                print(f"基因 [{gene_name}] 下载失败（第{retry + 1}次），{e}，10秒后重试...")
                time.sleep(10)  # 失败后延长休眠时间
            else:
                print(f"基因 [{gene_name}] 多次下载失败：{e}")
                return False


def batch_download_all_genes():
    """
    批量下载所有基因的详细数据（主函数）
    """
    # 1. 先获取所有去重基因列表
    gene_list = download_all_genes(SAVE_DIR)
    if not gene_list:
        print("未获取到基因列表，终止批量下载")
        return

    # 2. 创建基因详细数据的保存子目录
    gene_detail_dir = os.path.join(SAVE_DIR, "gene_details")
    os.makedirs(gene_detail_dir, exist_ok=True)

    # 3. 初始化失败日志
    with open(ERROR_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("以下基因下载失败：\n")

    # 4. 遍历下载所有基因
    total = len(gene_list)
    failed_genes = []

    for idx, gene in enumerate(gene_list, 1):
        print(f"\n📌 正在下载 [{idx}/{total}] 基因：{gene}")
        success = download_single_gene(gene, gene_detail_dir)
        if not success:
            failed_genes.append(gene)
            # 记录失败基因到日志（保留原始基因名，便于后续核对）
            with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{gene}\n")

    # 5. 下载完成总结
    print("\n" + "=" * 50)
    print(f"批量下载完成！")
    print(f"总基因数：{total}")
    print(f"成功数：{total - len(failed_genes)}")
    print(f"失败数：{len(failed_genes)}")
    if failed_genes:
        print(f"失败基因已记录至：{ERROR_LOG_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    batch_download_all_genes()