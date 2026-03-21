"""趋势分析模块 — 诊断历史持久化与统计。"""
import json
import os
import re
from collections import Counter
from config import HISTORY_FILE_PATH, HISTORY_MAX_RECORDS


def load_history() -> list[dict]:
    """从磁盘加载诊断历史记录。"""
    if not os.path.exists(HISTORY_FILE_PATH):
        return []
    try:
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(records: list[dict]):
    """将诊断历史写入磁盘（最多保留 HISTORY_MAX_RECORDS 条）。"""
    trimmed = records[-HISTORY_MAX_RECORDS:]
    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def extract_error_codes(text: str) -> list[str]:
    """正则提取日志文本中的故障代码（如 ERR_001, WARN_042, FAULT_12 等）。"""
    pattern = r"\b(?:ERR|WARN|FAULT|ALARM|ERROR|FAIL)[_-]?\d{1,5}\b"
    return re.findall(pattern, text, re.IGNORECASE)


def compute_error_frequency(records: list[dict], top_n: int = 20) -> list[tuple[str, int]]:
    """从历史记录中统计 Top N 故障代码频率。"""
    counter = Counter()
    for record in records:
        for item in record.get("results", []):
            codes = extract_error_codes(item.get("log", ""))
            counter.update(codes)
    return counter.most_common(top_n)


def compute_level_distribution(records: list[dict]) -> dict[str, int]:
    """统计所有历史记录中的风险等级分布。"""
    dist = {"严重": 0, "警告": 0, "正常": 0}
    for record in records:
        for item in record.get("results", []):
            level = item.get("level", "正常")
            if level in dist:
                dist[level] += 1
    return dist


def build_trend_prompt(error_freq: list[tuple[str, int]],
                       level_dist: dict[str, int]) -> str:
    """构建 AI 趋势分析 prompt。"""
    freq_text = "\n".join(f"  {code}: {count}次" for code, count in error_freq) if error_freq else "  暂无故障代码数据"
    dist_text = "\n".join(f"  {level}: {count}条" for level, count in level_dist.items())

    return f"""你是一位资深工业设备运维专家。请根据以下故障统计数据，给出趋势分析和预防建议。

## 故障代码频率 (Top 20)
{freq_text}

## 风险等级分布
{dist_text}

请从以下角度分析：
1. 高频故障模式识别：哪些故障代码出现频率异常？
2. 风险等级趋势：严重/警告/正常的比例是否健康？
3. 预防性维护建议：基于当前故障分布，给出具体的预防措施
4. 重点关注项：需要优先排查的设备或子系统

请用简洁专业的中文回答。"""
