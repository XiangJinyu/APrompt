# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph

import asyncio
import time
from typing import List, Literal

from pydantic import BaseModel, Field

from evaluator import DatasetType
from optimizer_utils.convergence_utils import ConvergenceUtils
from optimizer_utils.data_utils import DataUtils
from optimizer_utils.evaluation_utils import EvaluationUtils
from optimizer_utils.experience_utils import ExperienceUtils
from optimizer_utils.graph_utils import GraphUtils
from utils.logs import logger


QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 3  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_prompt())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            time.sleep(5)

    async def _optimize_prompt(self):

        prompt_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(prompt_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(prompt_path, self.round)
            # Load graph using graph_utils
            self.prompt = self.graph_utils.load_prompt(self.round, prompt_path)
            avg_score = await self.evaluation_utils.execute_prompt(self, directory, validation_n, data, initial=True)

        # Create a loop until the generated graph meets the check conditions

        directory = self.graph_utils.create_round_directory(prompt_path, self.round + 1)

        top_round = self.data_utils.get_best_round(self.sample)
        sample = top_round

        prompt = self.graph_utils.read_prompt_files(sample["round"], prompt_path)

        processed_experience = self.experience_utils.load_experience()
        experience = self.experience_utils.format_experience(processed_experience, sample["round"])

        log_data = self.data_utils.load_log(sample["round"])

        graph_optimize_prompt = self.graph_utils.create_prompt_optimize_prompt(
            experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
        )

        graph_optimize_node = await ActionNode.from_pydantic(GraphOptimize).fill(
            context=graph_optimize_prompt, mode="xml_fill", llm=self.optimize_llm
        )

        response = await self.graph_utils.get_prompt_optimize_response(graph_optimize_node)

        # Save the graph and evaluate
        self.graph_utils.write_prompt_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        self.graph = self.graph_utils.load_prompt(self.round + 1, prompt_path)

        logger.info(directory)

        avg_score = await self.evaluation_utils.execute_prompt(self, directory, validation_n, data, initial=False)

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    async def test(self):
        rounds = [5]  # You can choose the rounds you want to test here.
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.prompt = self.graph_utils.load_prompt(round, graph_path)

            score = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            new_data = self.data_utils.create_result_data(round, score)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)