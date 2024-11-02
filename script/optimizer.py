# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph

import asyncio
import time
from optimizer_utils.data_utils import DataUtils
from optimizer_utils.evaluation_utils import EvaluationUtils
from optimizer_utils.experience_utils import ExperienceUtils
from optimizer_utils.graph_utils import GraphUtils
from prompt.optimize_prompt import PROMPT_OPTIMIZE_PROMPT
from utils import load
from utils.logs import logger
from utils.response import responser, extract_content


class Optimizer:
    def __init__(
        self,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        name: str = "test"
    ) -> None:
        self.dataset = name
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)


    def optimize(self):

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(self._optimize_prompt())
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            time.sleep(5)

    async def _optimize_prompt(self):

        prompt_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(prompt_path)

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(prompt_path, self.round)
            # Load graph using graph_utils

            prompt, _, _ = load.load_meta_data()
            self.prompt = prompt
            self.graph_utils.write_prompt(directory, prompt=self.prompt)
            new_sample = await self.evaluation_utils.execute_prompt(self, directory, data, initial=True)
            _, answers = await self.evaluation_utils.evaluate_prompt(self, None, new_sample, path=prompt_path, data=data, initial=True)
            self.graph_utils.write_answers(directory, answers=answers)


        # Create a loop until the generated graph meets the check conditions

        _, requirements, qa = load.load_meta_data(3)

        directory = self.graph_utils.create_round_directory(prompt_path, self.round + 1)

        top_round = self.data_utils.get_best_round()

        sample = top_round

        prompt = sample['prompt']

        graph_optimize_prompt = PROMPT_OPTIMIZE_PROMPT.format(
            prompt=sample["prompt"], answers=sample["answers"],
            requirements=requirements,
            golden_answers=qa)

        response = await responser(messages=[{"role": "user", "content": graph_optimize_prompt}])

        modification = extract_content(response, "modification")
        prompt = extract_content(response, "prompt")

        self.prompt = prompt

        logger.info(directory)

        self.graph_utils.write_prompt(directory, prompt=self.prompt)

        new_sample = await self.evaluation_utils.execute_prompt(self, directory, data, initial=False)

        success, answers = await self.evaluation_utils.evaluate_prompt(self, sample, new_sample, path=prompt_path, data=data, initial=False)

        self.graph_utils.write_answers(directory, answers=answers)

        logger.info(prompt)
        logger.info(success)

        logger.info(self.round)

        return prompt
