import json
import os
import re
import time
import traceback
from typing import List
from utils.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_prompt(self, round_number: int, workflows_path: str):
        graph_file_name = f"{workflows_path}/prompt.txt"

        try:
            with open(graph_file_name, 'r', encoding='utf-8') as file:
                return file.read()  # 直接返回文本内容
        except FileNotFoundError as e:
            logger.info(f"Error loading graph for round {round_number}: {e}")
            raise

    def write_answers(self, directory: str, answers: dict):

        with open(os.path.join(directory, "answers.txt"), "w", encoding="utf-8") as file:
            for item in answers:
                file.write(f"问题: {item['question']}\n")
                file.write(f"回答: {item['answer']}\n")
                file.write("\n")

    def write_prompt(self, directory: str, prompt: str):

        with open(os.path.join(directory, "prompt.txt"), "w", encoding="utf-8") as file:
            file.write(prompt)
        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")


    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.info(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_prompt_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.info(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph = WORKFLOW_TEMPLATE.format(graph=response["graph"], round=round_number, dataset=dataset)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
