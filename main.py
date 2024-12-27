# -*- coding: utf-8 -*-
# @Date    : 10/11/2024 20:00 PM
# @Author  : Jinyu
# @Desc    : Entrance of APrompt.

from script.optimizer import Optimizer


if __name__ == "__main__":

    optimizer = Optimizer(
        optimized_path="workspace",
        initial_round=1,
        max_rounds=10,
        template="Navigate_paper.yaml",
        name="Navigate",
        optimize_model={"name": "gpt-4o-mini", "temperature": 0.7},
        # {"name": "claude-3-5-sonnet-20240620", "temperature": 0.7}
        execute_model={"name": "gpt-4o-mini", "temperature": 0},
        evaluate_model={"name": "gpt-4o-mini", "temperature": 0.3},
        iteration=True,
    )

    optimizer.optimize()