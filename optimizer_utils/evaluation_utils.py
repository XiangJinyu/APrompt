import asyncio

from script.evaluator import QuickEvaluate, QuickExecute
from utils.logs import logger


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def execute_prompt(self, optimizer, graph_path, data, model, initial=False, k=3):
        # 使用 optimizer 的 graph_utils 来加载图
        optimizer.prompt = optimizer.graph_utils.load_prompt(optimizer.round, graph_path)
        evaluator = QuickExecute(prompt=optimizer.prompt, k=k, model=model)

        # 使用 await 而不是 asyncio.run()
        answers = await evaluator.prompt_execute()

        cur_round = optimizer.round + 1 if not initial else optimizer.round

        new_data = {"round": cur_round, "answers": answers, "prompt": optimizer.prompt}

        return new_data

    async def evaluate_prompt(self, optimizer, sample, new_sample, path, data, model, initial=False):

        # 使用 optimizer 的 graph_utils 来加载图
        evaluator = QuickEvaluate(k=3)

        if initial is True:
            succeed = True
        else:
            # 连续执行三次评估
            evaluation_results = []
            for _ in range(4):
                result = await evaluator.prompt_evaluate(sample=sample, new_sample=new_sample, model=model)
                evaluation_results.append(result)

            logger.info(evaluation_results)

            true_count = evaluation_results.count(True)
            false_count = evaluation_results.count(False)
            succeed = true_count > false_count

        new_data = optimizer.data_utils.create_result_data(new_sample['round'], new_sample['answers'],
                                                           new_sample['prompt'], succeed)

        data.append(new_data)

        result_path = optimizer.data_utils.get_results_file_path(path)

        optimizer.data_utils.save_results(result_path, data)

        answers = new_sample['answers']

        return succeed, answers

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                self.root_path,
                is_test=False,
            )

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        return sum_score / validation_n

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            self.root_path,
            is_test=is_test,
        )
