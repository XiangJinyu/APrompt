# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets
import asyncio
from typing import Dict, Literal, Tuple, List, Any

from utils import load
from utils.response import responser, extract_content
from prompt.evaluate_prompt import EVALUATE_PROMPT


# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {"GSM8K": GSM8KBenchmark}

    async def graph_evaluate(
            self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = None  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"metagpt/ext/aflow/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"


class QuickExecute:
    """
    完成不同数据集的评估。
    """

    def __init__(self, prompt: str, k: int = 3, model: str = "gpt-4o-mini"):
        self.prompt = prompt
        self.k = k
        self.model = model

    async def prompt_evaluate(self) -> tuple[Any]:
        _, _, qa = load.load_meta_data(k=self.k)
        answers = []

        async def fetch_answer(q: str) -> Dict[str, Any]:
            messages = [{"role": "user", "content": self.prompt.format(question=q)}]
            try:
                answer = await responser(messages, model=self.model)  # 确保这是一个异步调用
                return {'question': q, 'answer': answer}
            except Exception as e:
                return {'question': q, 'answer': str(e)}

        tasks = [fetch_answer(item['question']) for item in qa]
        answers = await asyncio.gather(*tasks)

        return answers

class QuickEvaluate:
    """
    Complete the evaluation for different datasets here.
    """

    def __init__(self, k: int = 3):
        self.k = k

    async def prompt_evaluate(self, sample: list, new_sample: list) -> bool:
        _, requirement, qa = load.load_meta_data(k=self.k)

        messages = [{"role": "user", "content": EVALUATE_PROMPT.format(requirement=requirement, sample=sample, new_sample=new_sample, answers=str(qa))}]
        try:
            resp = await responser(messages)  # Assuming responser is an async function

            choose = extract_content(resp, 'choose')

            if choose == "B":
                return True
            else:
                return False

        except Exception as e:
            print(e)
            return False





if __name__ == "__main__":
    execute = QuickExecute(prompt="回答问题，{question}", k=3)

    # 使用asyncio.run来运行异步方法
    answers = asyncio.run(execute.prompt_evaluate())
    print(answers)
