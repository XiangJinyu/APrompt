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
        name="novel_content",
        optimize_model="claude-3-5-sonnet-20240620",
        execute_model="claude-3-5-sonnet-20240620",
        iteration=True,
    )

    optimizer.optimize()