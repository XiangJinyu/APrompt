import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from utils.logs import logger


class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_best_round(self, path=None, mode="Graph"):
        # 加载分数数据
        top_rounds = self._load_scores(path, mode)

        # 遍历已排序的top_scores，查找succeed为True的最大round
        for entry in self.top_scores:
            if entry["succeed"]:
                return entry  # 返回整个字典

        # 如果没有succeed为True的round，返回None或其他默认值
        return None

    def select_round(self, items):
        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        # 检查文件是否存在
        if not os.path.exists(log_dir):
            return ""  # 如果文件不存在，返回空字符串
        logger.info(log_dir)
        with open(log_dir, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, answers: list[dict], prompt: str, succeed: bool) -> dict:
        now = datetime.datetime.now()
        return {"round": round, "answers": answers, "prompt": prompt, "succeed": succeed, "time": now}

    def save_results(self, json_file_path: str, data: list):
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, default=str, indent=4)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        with open(result_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)

        # 遍历每个round，提取succeed字段
        for index, row in df.iterrows():
            self.top_scores.append({"round": row["round"], "succeed": row["succeed"], "prompt": row["prompt"], "answers":row['answers']})

        # 按round降序排序
        self.top_scores.sort(key=lambda x: x["round"], reverse=True)

        return self.top_scores

